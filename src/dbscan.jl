# DBSCAN Clustering
#
#   References:
#
#       Martin Ester, Hans-peter Kriegel, Jörg S, and Xiaowei Xu
#       A density-based algorithm for discovering clusters
#       in large spatial databases with noise. 1996.
#


type DbscanResult <: ClusteringResult
    seeds::Vector{Int}          # starting points of clusters, size (k,)
    assignments::Vector{Int}    # assignments, size (n,)
    counts::Vector{Int}         # number of points in each cluster, size (k,)
end


## main algorithm

function dbscan{T<:Real}(D::DenseMatrix{T}, eps::Real, minpts::Int)
    # check arguments
    n = size(D, 1)
    size(D, 2) == n || error("D must be a square matrix.")
    n >= 2 || error("There must be at least two points.")
    eps > 0 || error("eps must be a positive real value.")
    minpts >= 1 || error("minpts must be a positive integer.")

    # invoke core algorithm
    _dbscan(D, convert(T, eps), minpts, 1:n)
end

function _dbscan{T<:Real}(D::DenseMatrix{T}, eps::T, minpts::Int, visitseq::AbstractVector{Int})
    n = size(D, 1)

    # prepare
    seeds = Int[]
    counts = Int[]
    assignments = zeros(Int, n)
    visited = zeros(Bool, n)
    k = 0

    # main loop
    for p in visitseq
        if assignments[p] == 0 && !visited[p]
            visited[p] = true
            nbs = _dbs_region_query(D, p, eps)
            if length(nbs) >= minpts
                k += 1
                cnt = _dbs_expand_cluster!(D, k, p, nbs, eps, minpts, assignments, visited)
                push!(seeds, p)
                push!(counts, cnt)
            end
        end
    end

    # make output
    return DbscanResult(seeds, assignments, counts)
end

## key steps

function _dbs_region_query{T<:Real}(D::DenseMatrix{T}, p::Int, eps::T)
    n = size(D,1)
    nbs = Int[]
    dists = view(D,:,p)
    for i = 1:n
        @inbounds if dists[i] < eps
            push!(nbs, i)
        end
    end
    return nbs::Vector{Int}
end

function _dbs_expand_cluster!{T<:Real}(D::DenseMatrix{T},           # distance matrix
                                       k::Int,                      # the index of current cluster
                                       p::Int,                      # the index of seeding point
                                       nbs::Vector{Int},            # eps-neighborhood of p
                                       eps::T,                      # radius of neighborhood
                                       minpts::Int,                 # minimum number of neighbors of a density point
                                       assignments::Vector{Int},    # assignment vector
                                       visited::Vector{Bool})       # visited indicators
    assignments[p] = k
    cnt = 1
    while !isempty(nbs)
        q = shift!(nbs)
        if !visited[q]
            visited[q] = true
            qnbs = _dbs_region_query(D, q, eps)
            if length(qnbs) > minpts
                for x in qnbs
                    if assignments[x] == 0
                        push!(nbs, x)
                    end
                end
            end
        end
        if assignments[q] == 0
            assignments[q] = k
            cnt += 1
        end
    end
    return cnt
end


