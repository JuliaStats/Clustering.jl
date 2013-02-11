require("Options")
using OptionsMod

using Distance
using MLBase

###########################################################
#
# 	K-means options
#
###########################################################

type KmeansOpts
	max_iters::Int
	tol::RealValue
	weights::Union(Nothing, RealVec)	
end

function kmeans_opts(opts::Options)
	@defaults opts max_iter=200 tol=1.0e-6 
	@defaults opts weights=nothing
	
	kopts = KmeansOpts(max_iter, tol, weights)
	
	@check_used opts
	return kopts
end


###########################################################
#
# 	Core implementation
#
###########################################################

type KmeansProblem
	x::RealMat
	weights::Union(Nothing, RealVec)
end

type KmeansState
	centers::RealMat,
	assignments::Vector{Int},
	costs::RealVec,
	counts::Vector{Int},
	dmat::RealMat
end

type KmeansResult
	centers::RealMat,
	assignments::Vector{Int},
	costs::RealVec,
	counts::RealVec,
	total_cost::Real
end


function get_kmeans_result(pb::KmeansProblem, s::KmeansState)
	@assert pb.weights == nothing
	KmeansResult(s.centers, s.assignments, s.costs, s.counts, sum(s.costs))
end


function update_assignments!(dmat::RealMat, assignments::Vector{Int}, costs::RealVec, counts::Vector{Int})
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


function initialize_state(pb::KmeansProblem, centers::RealMat)
	n = size(pb.x, 2)
	k = size(init_centers, 2)
	
	dmat = pairwise(SqEuclidean(), centers, pb.x)
	
	costs = zero(eltype(dmat), n)
	assignments = Array(Int, n)
	counts = Array(Int, n)
	
	update_assignments!(dmat, assignments, costs, counts)
	KmeansState(centers, assignments, costs, counts, dmat)
end


function initialize_state(pb::KmeansProblem, k::Int)
	m = size(pb.x, 1)
	centers = zero(eltype(pb.x), (m, k))
	kmeanspp_initialize!(pb.x, centers)
	initialize_state(pb, centers)
end


function update_centers!(x::RealMat, assignments::Vector{Int}, costs::RealVec, counts::Vector{Int}, 
	centers::RealMat)
	
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
		c = counts[k]
		if c > 1  # nothing need to be done when c == 1
			inv_c = 1 / c
			@devec centers[:,i] .*= inv_c
		elseif c == 0
			add!(unused, i)
		end
	end
	
	if !isempty(unused)
		# pick new centers using a scheme like kmeans++
		
		ds = similar(costs)
		
		for i in unused
			tcosts = deepcopy(costs)
			j = sample_by_weights(tcosts)
			tcosts[j] = 0
			v = x[:,j]
			centers[:,i] = v
			
			ds = colwise!(ds, SqEuclidean(), v, x)
			tcosts = min(tcosts, ds)
		end
	end
end





function update!(pb::KmeansProblem, s::KmeansState)
	
	# update assignments of samples to centers
	update_assignments!(s.dmat, s.assignments, s.costs, s.counts)
	
	# update centers based on assignments
	update_centers!(pb.x, s.assignments, s.costs, s.counts, s.centers)
end


function _kmeans!(
	x::RealMat, 
	centers::RealMat, 
	assignments::RealVec,
	costs::RealVec,
	counts::Vector{Int},
	opts::KmeansOpts)
	
	iter_opts = iter_options(:minimize, opts.max_iters, opts.tol)
	mon = get_std_iter_monitor(opts.display)
	
	pb = KmeansProblem(x, nothing)
	state = init_state(pb, centers)
	
	iterative_update(pb, state, iter_opts, mon)	
	
	return get_kmeans_result(pb, state)
end

