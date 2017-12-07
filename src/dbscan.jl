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


immutable DbscanCluster <: ClusteringResult
    size::Int                      # number of points in cluster
    core_indices::Vector{Int}      # core points indices
    boundary_indices::Vector{Int}  # boundary points indices
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
            if length(qnbs) >= minpts
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

"""
    dbscan(points, radius ; leafsize = 20, min_neighbors = 1, min_cluster_size = 1) -> clusters

Cluster points using the DBSCAN (density-based spatial clustering of applications with noise) algorithm.

### Arguments
* `points`: matrix of points
* `radius::Real`: query radius

### Keyword Arguments
* `leafsize::Int`: number of points binned in each leaf node in the `KDTree`
* `min_neighbors::Int`: minimum number of neighbors to be a core point
* `min_cluster_size::Int`: minimum number of points to be a valid cluster

### Output
* `Vector{DbscanCluster}`: an array of clusters with the id, size core indices and boundary indices

### Example:
``` julia
points = randn(3, 10000)
clusters = dbscan(points, 0.05, min_neighbors = 3, min_cluster_size = 20) # clusters with less than 20 points will be discarded
```
"""
function dbscan(points::AbstractMatrix, radius::Real; leafsize::Int = 20, kwargs ...)
    kdtree = KDTree(points; leafsize=leafsize)
    return _dbscan(kdtree, points, radius; kwargs ...)
end


""" An implementation of DBSCAN algorithm that keeps track of both the core and boundary points """
function _dbscan(kdtree::KDTree, points::AbstractMatrix, radius::Real;
                 min_neighbors::Int = 1, min_cluster_size::Int = 1)
    dim, num_points = size(points)
    num_points <= dim && throw(ArgumentError("points has $dim rows and $num_points columns. Must be a D x N matric with D < N"))
    0 <= radius || throw(ArgumentError("radius $radius must be ≥ 0"))
    1 <= min_neighbors || throw(ArgumentError("min_neighbors $min_neighbors must be ≥ 1"))
    1 <= min_cluster_size || throw(ArgumentError("min_cluster_size $min_cluster_size must be ≥ 1"))

    clusters = Vector{DbscanCluster}(0)
    visited = falses(num_points)
    cluster_selection = falses(num_points)
    core_selection = falses(num_points)
    to_explore = Vector{Int}()
    adj_list = Vector{Int}()
    for i = 1:num_points
        visited[i] && continue
        push!(to_explore, i) # start a new cluster
        fill!(core_selection, false)
        fill!(cluster_selection, false)
        while !isempty(to_explore)
            current_index = shift!(to_explore)
            visited[current_index] && continue
            visited[current_index] = true
            append!(adj_list, inrange(kdtree, points[:, current_index], radius))
            cluster_selection[adj_list] = true
            # if a point doesn't have enough neighbors it is not a 'core' point and its neighbors are not added to the to_explore list
            if (length(adj_list) - 1) < min_neighbors
                empty!(adj_list)
                continue # query returns the query point as well as the neighbors
            end
            core_selection[current_index] = true
            update_exploration_list!(adj_list, to_explore, visited)
        end
        cluster_size = sum(cluster_selection)
        min_cluster_size <= cluster_size && accept_cluster!(clusters, core_selection, cluster_selection, cluster_size)
    end
    return clusters
end

"""
    update_exploration_list!(adj_list, exploration_list, visited)

Update the queue for expanding the cluster

### Input

* `adj_list :: Vector{Int}`: indices of the neighboring points
* `exploration_list :: Vector{Int}`: the indices that  will be explored in the future
* `visited :: Vector{Bool}`: a flag to indicate whether a point has been explored already
"""
function update_exploration_list!{T <: Int}(adj_list::Array{T}, exploration_list::Vector{T}, visited::BitArray{1})
    for j in adj_list
        visited[j] && continue
        push!(exploration_list, j)
    end
    empty!(adj_list)
end

"""
    accept_cluster!(clusters, core_selection, cluster_selection)

Accept cluster and update the clusters list

### Input

* `clusters :: Vector{DbscanCluster}`: a list of the accepted clusters
* `core_selection :: Vector{Bool}`: selection of the core points of the cluster
* `cluster_selection :: Vector{Bool}`: selection of all the cluster points
"""
function accept_cluster!(clusters::Vector{DbscanCluster}, core_selection::BitVector,
                         cluster_selection::BitVector, cluster_size::Int)
    core_idx = find(core_selection) # index list of the core members
    boundary_selection = cluster_selection .& (~).(core_selection) #TODO change to .~ core_selection
                                                                            # when dropping 0.5
    boundary_idx = find(boundary_selection) # index list of the boundary members
    push!(clusters, DbscanCluster(cluster_size, core_idx, boundary_idx))
end
