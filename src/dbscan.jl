# DBSCAN Clustering
#
#   References:
#
#       Martin Ester, Hans-peter Kriegel, Jörg S, and Xiaowei Xu
#       A density-based algorithm for discovering clusters
#       in large spatial databases with noise. 1996.
#

"""
The output of [`dbscan`](@ref) function (distance matrix-based implementation).

## Fields
 - `seeds::Vector{Int}`: indices of cluster starting points
 - `assignments::Vector{Int}`: vector of clusters indices, where each point was assigned to
 - `counts::Vector{Int}`: cluster sizes (number of assigned points)
end
```
"""
mutable struct DbscanResult <: ClusteringResult
    seeds::Vector{Int}          # starting points of clusters, size (k,)
    assignments::Vector{Int}    # assignments, size (n,)
    counts::Vector{Int}         # number of points in each cluster, size (k,)
end

"""
DBSCAN cluster returned by [`dbscan`](@ref) function (point coordinates-based
implementation)

## Fields
 * `size::Int`: number of points in a cluster (core + boundary)
 * `core_indices::Vector{Int}`: indices of points in the cluster *core*
 * `boundary_indices::Vector{Int}`: indices of points on the cluster *boundary*
"""
struct DbscanCluster <: ClusteringResult
    size::Int                      # number of points in cluster
    core_indices::Vector{Int}      # core points indices
    boundary_indices::Vector{Int}  # boundary points indices
end

## main algorithm

"""
    dbscan(D::DenseMatrix, eps::Real, minpts::Int)

Perform DBSCAN algorithm using the distance matrix `D`.

Returns an instance of [`DbscanResult`](@ref).

# Algorithm Options
The following options control which points would be considered
*density reachable*:
  - `eps::Real`: the radius of a point neighborhood
  - `minpts::Int`: the minimum number of neighboring points (including itself)
     to qualify a point as a density point.
"""
function dbscan(D::DenseMatrix{T}, eps::Real, minpts::Int) where T<:Real
    # check arguments
    n = size(D, 1)
    size(D, 2) == n || error("D must be a square matrix.")
    n >= 2 || error("There must be at least two points.")
    eps > 0 || error("eps must be a positive real value.")
    minpts >= 1 || error("minpts must be a positive integer.")

    # invoke core algorithm
    _dbscan(D, convert(T, eps), minpts, 1:n)
end

function _dbscan(D::DenseMatrix{T}, eps::T, minpts::Int, visitseq::AbstractVector{Int}) where T<:Real
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

function _dbs_region_query(D::DenseMatrix{T}, p::Int, eps::T) where T<:Real
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

function _dbs_expand_cluster!(D::DenseMatrix{T},           # distance matrix
                              k::Int,                      # the index of current cluster
                              p::Int,                      # the index of seeding point
                              nbs::Vector{Int},            # eps-neighborhood of p
                              eps::T,                      # radius of neighborhood
                              minpts::Int,                 # minimum number of neighbors of a density point
                              assignments::Vector{Int},    # assignment vector
                              visited::Vector{Bool}) where T<:Real       # visited indicators
    assignments[p] = k
    cnt = 1
    while !isempty(nbs)
        q = popfirst!(nbs)
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
    dbscan(points::AbstractMatrix, radius::Real;
           leafsize = 20, min_neighbors = 1, min_cluster_size = 1)

Cluster `points` using the DBSCAN (density-based spatial clustering of
applications with noise) algorithm.

Returns the clustering as a vector of [`DbscanCluster`](@ref) objects.

### Arguments
 - `points`: the ``d×n`` matrix of points. `points[:, j]` is a
   ``d``-dimensional coordinates of ``j``-th point
 - `radius::Real`: query radius

Additional keyword options to control the algorithm:
 - `leafsize::Int` (defaults to 20): the number of points binned in each
   leaf node in the `KDTree`
 - `min_neighbors::Int` (defaults to 1): the minimum number of a *core* point
   neighbors
 - `min_cluster_size::Int` (defaults to 1): the minimum number of points in
   a valid cluster

### Example:
``` julia
points = randn(3, 10000)
# DBSCAN clustering, clusters with less than 20 points will be discarded:
clusters = dbscan(points, 0.05, min_neighbors = 3, min_cluster_size = 20)
```
"""
function dbscan(points::AbstractMatrix, radius::Real; leafsize::Int = 20, kwargs ...)
    kdtree = KDTree(points; leafsize=leafsize)
    return _dbscan(kdtree, points, radius; kwargs ...)
end


# An implementation of DBSCAN algorithm that keeps track of both the core and boundary points
function _dbscan(kdtree::KDTree, points::AbstractMatrix, radius::Real;
                 min_neighbors::Int = 1, min_cluster_size::Int = 1)
    dim, num_points = size(points)
    num_points <= dim && throw(ArgumentError("points has $dim rows and $num_points columns. Must be a D x N matric with D < N"))
    0 <= radius || throw(ArgumentError("radius $radius must be ≥ 0"))
    1 <= min_neighbors || throw(ArgumentError("min_neighbors $min_neighbors must be ≥ 1"))
    1 <= min_cluster_size || throw(ArgumentError("min_cluster_size $min_cluster_size must be ≥ 1"))

    clusters = Vector{DbscanCluster}()
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
            current_index = popfirst!(to_explore)
            visited[current_index] && continue
            visited[current_index] = true
            append!(adj_list, inrange(kdtree, points[:, current_index], radius))
            cluster_selection[adj_list] .= true
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
function update_exploration_list!(adj_list::Array{T}, exploration_list::Vector{T}, visited::BitArray{1}) where T <: Int
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
    core_idx = findall(core_selection) # index list of the core members
    boundary_selection = cluster_selection .& (~).(core_selection) #TODO change to .~ core_selection
                                                                            # when dropping 0.5
    boundary_idx = findall(boundary_selection) # index list of the boundary members
    push!(clusters, DbscanCluster(cluster_size, core_idx, boundary_idx))
end
