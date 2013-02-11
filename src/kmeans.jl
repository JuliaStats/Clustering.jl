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


function init_assignments!(dmat::FPMat, assignments::Vector{Int}, costs::FPVec, counts::Vector{Int})
	
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

function update_assignments!(dmat::FPMat, assignments::Vector{Int}, 
	costs::FPVec, counts::Vector{Int}, affected::BitVector)
	
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
		
		pa = assignments[j]
		if pa != a
			assignments[j] = a
			affected[a] = true
			affected[pa] = true
		end	
		costs[j] = c
		counts[a] += 1
	end
	
	# everything that gets zero count is also tagged as affected
	for i = 1 : k
		if counts[i] == 0
			affected[i] = true
		end
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


function update_centers!(x::FPMat, w::Nothing, assignments::Vector{Int}, 
	costs::FPVec, counts::Vector{Int}, centers::FPMat, affected::BitVector)
	
	n = size(x, 2)
	k = size(centers, 2)
	
	s = falses(k)
	for j = 1 : n
		i = assignments[j]
		if affected[i]
			if s[i]
				@devec centers[:,i] += x[:,j]
			else
				@devec centers[:,i] = x[:,j]
				s[i] = true
			end
		end
	end
	
	unused = IntSet()
		
	for i = 1 : k
		if affected[i]
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
	end
	
	if !isempty(unused)
		repick_unused_centers(x, costs, centers, unused)
	end
end


function update_centers!(x::FPMat, weights::FPVec, assignments::Vector{Int}, 
	costs::FPVec, counts::Vector{Int}, centers::FPMat, affected::BitVector)
	
	n = size(x, 2)
	k = size(centers, 2)
	
	s = zeros(eltype(weights), k)
	for j = 1 : n
		i = assignments[j]
		if affected[i]
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
	end
	
	unused = IntSet()
		
	for i = 1 : k
		if affected[i]
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
	iterations::Int
	converged::Bool
end

# core k-means skeleton

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
	
	k = size(centers, 2)
	affected = trues(k) # indicators of whether a center needs to be updated
	num_affected = k
	
	dmat = pairwise(SqEuclidean(), centers, x)
	init_assignments!(dmat, assignments, costs, counts)
	objv = w == nothing ? sum(costs) : dot(w, costs)
	
	# main loop
	if disp_level >= 2
		@printf "%7s %18s %18s | %8s \n" "Iters" "objv" "objv-change" "affected"
		println("-------------------------------------------------------------")
	end
	
	t = 0
	
	converged = false
	
	while !converged && t < opts.max_iters
		t = t + 1
		
		# update (affected) centers
		
		update_centers!(x, w, assignments, costs, counts, centers, affected)
		
		# update pairwise distance matrix
		
		if t == 1 || num_affected > 0.75 * k  
			pairwise!(dmat, SqEuclidean(), centers, x)
		else
			# if only a small subset is affected, only compute for that subset
			affected_inds = find(affected)
			dmat_p = pairwise(SqEuclidean(), centers[:, affected_inds], x)
			dmat[affected_inds, :] = dmat_p
		end
		
		# update assignments
		
		fill!(affected, false)
		update_assignments!(dmat, assignments, costs, counts, affected)		
		num_affected = sum(affected)
		
		# compute change of objective and determine convergence
		
		prev_objv = objv
		objv = w == nothing ? sum(costs) : dot(w, costs)
		objv_change = objv - prev_objv 
		
		if objv_change > tol
			warn("The objective value changes towards an opposite direction")
		end
		
		if abs(objv_change) < tol
			converged = true
		end
		
		# display iteration information (if asked)
			
		if disp_level >= 2
			@printf "%7d %18.6e %18.6e | %8d\n" t objv objv_change num_affected 
		end
	end
	
	if disp_level >= 1
		if converged
			println("K-means converged with $t iterations (objv = $objv)")
		else
			println("K-means terminated without convergence after $t iterations (objv = $objv)")
		end
	end
	
	return KmeansResult(centers, assignments, costs, counts, objv, t, converged)
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
	
	assignments = zeros(Int, n)
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


