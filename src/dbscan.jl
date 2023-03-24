# DBSCAN Clustering
#


"""
    DbscanCluster

DBSCAN cluster, part of [`DbscanResult`](@ref) returned by [`dbscan`](@ref) function.

## Fields
  - `size::Int`: number of points in a cluster (core + boundary)
  - `core_indices::Vector{Int}`: indices of points in the cluster *core*, a.k.a. *seeds*
     (have at least `min_neighbors` neighbors in the cluster)
  - `boundary_indices::Vector{Int}`: indices of the cluster points outside of *core*
"""
struct DbscanCluster
    size::Int
    core_indices::Vector{Int}
    boundary_indices::Vector{Int}
end

"""
    DbscanResult <: ClusteringResult

The output of [`dbscan`](@ref) function.

## Fields
  - `clusters::Vector{DbscanCluster}`: clusters, length *K*
  - `seeds::Vector{Int}`: indices of the first points of each cluster's *core*, length *K*
  - `counts::Vector{Int}`: cluster sizes (number of assigned points), length *K*
  - `assignments::Vector{Int}`: vector of clusters indices, where each point was assigned to, length *N*
"""
struct DbscanResult <: ClusteringResult
    clusters::Vector{DbscanCluster}
    seeds::Vector{Int}
    counts::Vector{Int}
    assignments::Vector{Int}

    function DbscanResult(clusters::AbstractVector{DbscanCluster}, num_points::Integer)
        assignments = zeros(Int, num_points)
        for (i, clu) in enumerate(clusters)
            assignments[clu.core_indices] .= i
            assignments[clu.boundary_indices] .= i
        end
        new(clusters,
            [c.core_indices[1] for c in clusters],
            [c.size for c in clusters],
            assignments)
    end
end


"""
    dbscan(points::AbstractMatrix, radius::Real;
           [metric=Euclidean()],
           [min_neighbors=1], [min_cluster_size=1],
           [nntree_kwargs...]) -> DbscanResult

Cluster `points` using the DBSCAN (Density-Based Spatial Clustering of
Applications with Noise) algorithm.

## Arguments
 - `points`: when `metric` is specified, the *d×n* matrix, where
   each column is a *d*-dimensional coordinate of a point;
   when `metric=nothing`, the *n×n* matrix of pairwise distances between the points
 - `radius::Real`: neighborhood radius; points within this distance
   are considered neighbors

Optional keyword arguments to control the algorithm:
 - `metric` (defaults to `Euclidean()`): the points distance metric to use,
   `nothing` means `points` is the *n×n* precalculated distance matrix
 - `min_neighbors::Integer` (defaults to 1): the minimal number of neighbors
   required to assign a point to a cluster "core"
 - `min_cluster_size::Integer` (defaults to 1): the minimal number of points in
   a cluster; cluster candidates with fewer points are discarded
 - `nntree_kwargs...`: parameters (like `leafsize`) for the `KDTree` constructor

## Example
```julia
points = randn(3, 10000)
# DBSCAN clustering, clusters with less than 20 points will be discarded:
clustering = dbscan(points, 0.05, min_neighbors = 3, min_cluster_size = 20)
```

## References:

  * Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu,
    *"A density-based algorithm for discovering clusters
    in large spatial databases with noise"*, KDD-1996, pp. 226--231.
  * Erich Schubert, Jörg Sander, Martin Ester, Hans Peter Kriegel, and
    Xiaowei Xu, *"DBSCAN Revisited, Revisited: Why and How You Should
    (Still) Use DBSCAN"*, ACM Transactions on Database Systems,
    Vol.42(3)3, pp. 1--21, https://doi.org/10.1145/3068335
"""
function dbscan(points::AbstractMatrix, radius::Real;
                metric = Euclidean(),
                min_neighbors::Integer = 1, min_cluster_size::Integer = 1,
                nntree_kwargs...)
    0 <= radius || throw(ArgumentError("radius $radius must be ≥ 0"))

    if metric !== nothing
        # points are point coordinates
        dim, num_points = size(points)
        num_points <= dim && throw(ArgumentError("points has $dim rows and $num_points columns. Must be a D x N matric with D < N"))
        kdtree = KDTree(points, metric; nntree_kwargs...)
        data = (kdtree, points)
    else
        # points is a distance matrix
        num_points = size(points, 1)
        size(points, 2) == num_points || throw(ArgumentError("When metric=nothing, points must be a square distance matrix ($(size(points)) given)."))
        num_points >= 2 || throw(ArgumentError("At least two data points are required ($num_points given)."))
        data = points
    end
    clusters = _dbscan(data, num_points, radius, min_neighbors, min_cluster_size)
    return DbscanResult(clusters, num_points)
end

# An implementation of DBSCAN algorithm that keeps track of both the core and boundary points
function _dbscan(data::Union{AbstractMatrix, Tuple{NNTree, AbstractMatrix}},
                 num_points::Integer, radius::Real,
                 min_neighbors::Integer, min_cluster_size::Integer)
    1 <= min_neighbors || throw(ArgumentError("min_neighbors $min_neighbors must be ≥ 1"))
    1 <= min_cluster_size || throw(ArgumentError("min_cluster_size $min_cluster_size must be ≥ 1"))

    clusters = Vector{DbscanCluster}()
    visited = fill(false, num_points)
    cluster_mask = Vector{Bool}(undef, num_points)
    core_mask = similar(cluster_mask)
    to_explore = Vector{Int}()
    neighbors = Vector{Int}()
    @inbounds for i = 1:num_points
        visited[i] && continue
        @assert isempty(to_explore)
        push!(to_explore, i) # start a new cluster
        fill!(core_mask, false)
        fill!(cluster_mask, false)
        # depth-first search to find all points in the cluster
        while !isempty(to_explore)
            point = popfirst!(to_explore)
            visited[point] && continue
            visited[point] = true
            _dbscan_region_query!(neighbors, data, point, radius)
            cluster_mask[neighbors] .= true # mark as candidates

            # if a point has enough neighbors, it is a 'core' point and its neighbors are added to the to_explore list
            if length(neighbors) >= min_neighbors
                core_mask[point] = true
                for j in neighbors
                    visited[j] || push!(to_explore, j)
                end
            end
            empty!(neighbors)
        end

        # if the cluster has core and is large enough, it is accepted
        if any(core_mask) && (cluster_size = sum(cluster_mask)) >= min_cluster_size
            core = Vector{Int}()
            boundary = Vector{Int}()
            for (i, (is_cluster, is_core)) in enumerate(zip(cluster_mask, core_mask))
                @assert is_core && is_cluster || !is_core # core is always in a cluster
                is_cluster && push!(ifelse(is_core, core, boundary), i)
            end
            @assert !isempty(core)
            push!(clusters, DbscanCluster(cluster_size, core, boundary))
        end
    end
    return clusters
end

# distance matrix-based
function _dbscan_region_query!(neighbors::AbstractVector, dists::AbstractMatrix,
                               point::Integer, radius::Real)
    empty!(neighbors)
    for (i, dist) in enumerate(view(dists, :, point))
        (dist <= radius) && push!(neighbors, i)
    end
    return neighbors
end

# NN-tree based
function _dbscan_region_query!(neighbors::AbstractVector,
                               nntree_and_points::Tuple{NNTree, AbstractMatrix},
                               point::Integer, radius::Real)
    nntree, points = nntree_and_points
    empty!(neighbors)
    return append!(neighbors, inrange(nntree, view(points, :, point), radius))
end
