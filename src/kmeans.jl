require("Options")
using OptionsMod

using Distance
using Devectorize
using MLBase

###########################################################
#
# 	K-means options
#
###########################################################

type KmeansOpts
	max_iters::Int
	tol::Real
	weights::Union(Nothing, FPVec)	
	display::Symbol
end


function kmeans_opts(opts::Options)
	@defaults opts max_iter=200 tol=1.0e-6 
	@defaults opts weights=nothing
	@defaults opts display=:iter
		
	kopts = KmeansOpts(max_iter, tol, weights, display)
	
	@check_used opts
	return kopts
end

function kmeans_opts()
	o = @options
	kmeans_opts(o)
end


###########################################################
#
# 	Core implementation
#
###########################################################


function update_assignments!(dmat::FPMat, assignments::Vector{Int}, costs::FPVec, counts::Vector{Int})
	k, n = size(dmat)
	fill!(counts, 0)
	
	for j = 1 : n
		a = 1
		c = dmat[1, j]
		for i = 2 : k
			ci = dmat[i, j]
			if ci < c
				a = i
				c = ci
			end
		end

		assignments[j] = a		
		costs[j] = c
		counts[a] += 1
	end
end


function repick_unused_centers(x::FPMat, costs::FPVec, centers::FPMat, unused::IntSet)
	# pick new centers using a scheme like kmeans++		
	ds = similar(costs)
	tcosts = deepcopy(costs)
		
	for i in unused
		j = sample_by_weights(tcosts)
		tcosts[j] = 0
		v = x[:,j]
		centers[:,i] = v
			
		colwise!(ds, SqEuclidean(), v, x)
		tcosts = min(tcosts, ds)
	end
end


function update_centers!(x::FPMat, w::Nothing, 
	assignments::Vector{Int}, costs::FPVec, counts::Vector{Int}, centers::FPMat)
	
	n = size(x, 2)
	k = size(centers, 2)
	
	s = falses(k)
	for j = 1 : n
		i = assignments[j]
		if s[i]
			@devec centers[:,i] += x[:,j]
		else
			@devec centers[:,i] = x[:,j]
			s[i] = true
		end
	end
	
	unused = IntSet()
		
	for i = 1 : k
		c = counts[i]
		if c > 0  
			if c != 1 # nothing need to be done when c == 1
				inv_c = 1 / c
				@devec centers[:,i] .*= inv_c
			end
		else
			add!(unused, i)
		end
	end
	
	if !isempty(unused)
		repick_unused_centers(x, costs, centers, unused)
	end
end


function update_centers!(x::FPMat, weights::FPVec, 
	assignments::Vector{Int}, costs::FPVec, counts::Vector{Int}, centers::FPMat)
	
	n = size(x, 2)
	k = size(centers, 2)
	
	s = zeros(eltype(weights), k)
	for j = 1 : n
		i = assignments[j]
		wj = weights[j]
		@assert wj >= 0
		
		if wj > 0
			if s[i] > 0
				@devec centers[:,i] += wj .* x[:,j]
			else
				@devec centers[:,i] = wj .* x[:,j]
			end
			s[i] += wj
		end
	end
	
	unused = IntSet()
		
	for i = 1 : k
		c = s[i]
		if c > 0  
			if c != 1 # nothing need to be done when c == 1
				inv_c = 1 / c
				@devec centers[:,i] .*= inv_c
			end
		else
			add!(unused, i)
		end
	end
	
	if !isempty(unused)
		repick_unused_centers(x, costs, centers, unused)
	end
end


type KmeansResult
	centers::FPMat
	assignments::Vector{Int}
	costs::FPVec
	counts::Vector{Int}
	total_cost::Real
end

function _kmeans!(
	x::FPMat, 
	centers::FPMat, 
	assignments::Vector{Int},
	costs::FPVec,
	counts::Vector{Int},
	opts::KmeansOpts)

	# process options

	tol = opts.tol
	w = opts.weights
	
	disp_level = 
		opts.display == :none ? 0 :
		opts.display == :final ? 1 :
		opts.display == :iter ? 2 :
		throw(ArgumentError("Invalid value for the option 'display'.")) 

	# initialize	
	dmat = pairwise(SqEuclidean(), centers, x)
	update_assignments!(dmat, assignments, costs, counts)
	objv = w == nothing ? sum(costs) : dot(w, costs)
	
	# main loop
	
	if disp_level >= 2
		@printf "%6s %18s %18s\n" "Iters" "objv" "objv-change"
	end
	
	t = 0
	converged = false
	
	while !converged && t < opts.max_iters
		t = t + 1
		
		update_centers!(x, w, assignments, costs, counts, centers)
		pairwise!(dmat, SqEuclidean(), centers, x)
		
		update_assignments!(dmat, assignments, costs, counts)
		
		prev_objv = objv
		objv = w == nothing ? sum(costs) : dot(w, costs)
		objv_change = objv - prev_objv 
		
		if objv_change > tol
			warn("The objective value changes towards an opposite direction")
		end
		
		if abs(objv_change) < tol
			converged = true
		end
			
		if disp_level >= 2
			@printf "%5d: %18.6e %18.6e\n" t objv objv_change 
		end
	end
	
	if disp_level >= 1
		if converged
			println("K-means converged with $t iterations (objv = $objv)")
		else
			println("K-means terminated without convergence after $t iterations (objv = $objv)")
		end
	end
	
	return KmeansResult(centers, assignments, costs, counts, objv)
end


###########################################################
#
# 	Interface functions
#
###########################################################

function check_k(n, k)
	if !(k >=2 && k < n)
		throw( ArgumentError("k must be in [2, n)") )
	end
end

function kmeans!(x::FPMat, centers::FPMat, opts::KmeansOpts)
	m, n = size(x)
	m2, k = size(centers)
	if m != m2
		throw(ArgumentError("Mismatched dimensions in x and init_centers."))
	end
	check_k(n, k)
	
	w = opts.weights
	if w != nothing
		if length(w) != size(x, 2)
			throw(ArgumentError("The lenght of w must match the number of columns in x."))
		end
	end
	
	assignments = Array(Int, n)
	costs = zeros(n)
	counts = Array(Int, k)
	
	_kmeans!(x, centers, assignments, costs, counts, opts)
end


function kmeans(x::FPMat, init_centers::FPMat, opts::KmeansOpts)
	centers = deepcopy(init_centers)
	kmeans!(x, centers, opts)
end

kmeans(x::FPMat, init_centers::FPMat, opts::Options) = kmeans(x, init_centers, kmeans_opts(opts))
kmeans(x::FPMat, init_centers::FPMat) = kmeans(x, init_centers, kmeans_opts())


function kmeans(x::FPMat, k::Int, opts::KmeansOpts)
	m, n = size(x)
	check_k(n, k)
	init_centers = Array(eltype(x), (m, k)) 
	kmeanspp_initialize!(x, init_centers)
	kmeans!(x, init_centers, opts)
end

kmeans(x::FPMat, k::Int, opts::Options) = kmeans(x, k, kmeans_opts(opts))
kmeans(x::FPMat, k::Int) = kmeans(x, k, kmeans_opts())


