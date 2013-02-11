# Functions for choosing initial centers (often called seeds)

using MLBase
using Distance


function randseed_initialize!(x::FPMat, centers::FPMat)
	n = size(x, 2)
	k = size(centers, 2)
	si = sample_without_replacement(1:n, k)
	centers[:,:] = x[:,si]
end

function kmeanspp_initialize!(x::FPMat, centers::FPMat)
	n = size(x, 2)
	k = size(centers, 2)
	
	# randomly pick the first center
	si = rand(1:n)
	v = x[:,si]
	centers[:,1] = v
	
	# initialize the cost vector
	costs = colwise(SqEuclidean(), v, x)
	
	# pick remaining (with a chance proportional to cost)
	for i = 2 : k
		si = sample_by_weights(costs)
		v = x[:,si]
		centers[:,i] = v
		
		# update costs
		
		if i < k
			new_costs = colwise(SqEuclidean(), v, x)
			costs = min(costs, new_costs)
		end
	end
end


