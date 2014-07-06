# Functions for choosing initial centers (often called seeds)

function randseed_initialize!{T<:FloatingPoint}(x::Matrix{T}, centers::Matrix{T})
    n = size(x, 2)
    k = size(centers, 2)
    si = sample(1:n, k, replace=false)
    centers[:,:] = x[:,si]
end

function kmeanspp_initialize!{T<:FloatingPoint}(x::Matrix{T}, centers::Matrix{T})
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
        si = wsample(1:n, costs)
        v = x[:,si]
        centers[:,i] = v

        # update costs

        if i < k
            new_costs = colwise(SqEuclidean(), v, x)
            costs = min(costs, new_costs)
        end
    end
end

# How good a center for the whole dataset is the j'th element? Used for
# initial_medoids.
function kmedoids_centrality{R <: FloatingPoint}(dist::Matrix{R}, j::Int, denoms::Vector{R})
    n = size(dist)[1]

    # Rolling our own sum, because the built-in sum() function is slower and
    # this turns out to be a bottleneck for the whole algorithm.
    # This may change if/when the compiler gains the ability to inline
    # function arguments.
    c::R = 0.0
    for i = 1:n
        c += dist[i,j]/denoms[i]
    end
    c
end

# Calculate a set of k initial medoids using the algorithm given in "A simple
# and fast algorithm for K-medoids clustering" by Hae-Sang Park and Chi-Hyuck
# Jun, doi:10.1016/j.eswa.2008.01.039
# Accepts a symmetric matrix of distances and k; returns a vector of indices
# into the dataset.
function initial_medoids{R <: FloatingPoint}(dist::Matrix{R}, k::Int)
    n = size(dist)[1]
    denoms::Vector{R} = sum(dist, 2)[:]

    scores::Vector{(Float64, Int)} = sort([(kmedoids_centrality(dist, i, denoms), i) for i = 1:n])

    [s[2] for s in scores[1:k]]::Vector{Int}
end
