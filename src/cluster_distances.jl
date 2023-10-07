#===
Base type for efficient computation of average(mean) distances
from the cluster centers to a given point.

The descendant types should implement the following methods:
  * `update!(dists, assignments, points)`: update the internal
    state of `dists` with point coordinates and their assignments to the clusters
  * `sumdistances(dists, points, indices)`: compute the sum of
    distances from all `dists` clusters to `points`
===#
abstract type ClusterDistances{T} end

# create empty ClusterDistances object for a given metric
# and update it with a given clustering
# if batch_size is specified, the updates are done in point batches of given size
function ClusterDistances(metric::SemiMetric,
                 assignments::AbstractVector{<:Integer},
                 points::AbstractMatrix{<:Real},
                 batch_size::Union{Integer, Nothing} = nothing)
    update!(ClusterDistances(eltype(points), metric, length(assignments), size(points, 1),
                             maximum(assignments)),
            assignments, points, batch_size)
end

ClusterDistances(metric, R::ClusteringResult, args...) =
    ClusterDistances(metric, assignments(R), args...)

# fallback implementations of ClusteringDistances methods

cluster_sizes(dists::ClusterDistances) = dists.cluster_sizes
nclusters(dists::ClusterDistances) = length(cluster_sizes(dists))

update!(dists::ClusterDistances,
        assignments::AbstractVector, points::AbstractMatrix) =
    error("update!(dists::$(typeof(dists))) not implemented")

sumdistances(dists::ClusterDistances,
             points::Union{AbstractMatrix, Nothing},
             indices::Union{AbstractVector{<:Integer}, Nothing}) =
    error("sumdistances(dists::$(typeof(dists))) not implemented")

# average distances from each cluster to each point, nclusters×n matrix
function meandistances(dists::ClusterDistances,
                       assignments::AbstractVector{<:Integer},
                       points::Union{AbstractMatrix, Nothing},
                       indices::AbstractVector{<:Integer})
    @assert length(assignments) == length(indices)
    (points === nothing) || @assert(size(points, 2) == length(indices))
    clu_to_pt = sumdistances(dists, points, indices)
    clu_sizes = cluster_sizes(dists)
    @assert length(assignments) == length(indices)
    @assert size(clu_to_pt) == (length(clu_sizes), length(assignments))

    # normalize distances by cluster sizes
    @inbounds for j in eachindex(assignments)
        for (i, c) in enumerate(clu_sizes)
            if i == assignments[j]
                c -= 1
            end
            if c == 0
                clu_to_pt[i,j] = 0
            else
                clu_to_pt[i,j] /= c
            end
        end
    end
    return clu_to_pt
end

# wrapper for ClusteringResult
update!(dists::ClusterDistances, R::ClusteringResult, args...) =
    update!(dists, assignments(R), args...)

# batch-update silhouette dists (splitting the points into chunks of batch_size size)
function update!(dists::ClusterDistances,
                 assignments::AbstractVector{<:Integer}, points::AbstractMatrix{<:Real},
                 batch_size::Union{Integer, Nothing})
    n = size(points, 2)
    ((batch_size === nothing) || (n <= batch_size)) &&
        return update!(dists, assignments, points)

    for batch_start in 1:batch_size:n
        batch_ixs = batch_start:min(batch_start + batch_size - 1, n)
        update!(dists, view(assignments, batch_ixs), view(points, :, batch_ixs))
    end
    return dists
end

# generic ClusterDistances implementation for an arbitrary metric M
# if M is Nothing, point_dists is an arbitrary matrix of point distances
struct SimpleClusterDistances{M, T} <: ClusterDistances{T}
    metric::M
    cluster_sizes::Vector{Int}
    assignments::Vector{Int}
    point_dists::Matrix{T}

    SimpleClusterDistances(::Type{T}, metric::M,
                           npoints::Integer, nclusters::Integer) where {M<:Union{SemiMetric, Nothing}, T<:Real} =
        new{M, T}(metric, zeros(Int, nclusters), Vector{Int}(),
                  Matrix{T}(undef, npoints, npoints))

    # reuse given points matrix
    function SimpleClusterDistances(
        metric::Nothing,
        assignments::AbstractVector{<:Integer},
        point_dists::AbstractMatrix{T}
    ) where T<:Real
        n = length(assignments)
        size(point_dists) == (n, n) || throw(DimensionMismatch("assignments length ($n) does not match distances matrix size ($(size(point_dists)))"))
        issymmetric(point_dists) || throw(ArgumentError("point distances matrix must be symmetric"))
        clu_sizes = zeros(Int, maximum(assignments))
        @inbounds for cluster in assignments
            clu_sizes[cluster] += 1
        end
        new{Nothing, T}(metric, clu_sizes, assignments, point_dists)
    end
end

# fallback ClusterDistances constructor
ClusterDistances(::Type{T}, metric::Union{SemiMetric, Nothing},
                 npoints::Union{Integer, Nothing}, dims::Integer, nclusters::Integer) where T<:Real =
    SimpleClusterDistances(T, metric, npoints, nclusters)

# when metric is nothing, points is the matrix of distances
function ClusterDistances(metric::Nothing,
                          assignments::AbstractVector{<:Integer},
                          points::AbstractMatrix,
                          batch_size::Union{Integer, Nothing} = nothing)
    (batch_size === nothing) || (size(points, 2) > batch_size) ||
        error("batch-updates of distance matrix-based ClusterDistances not supported")
    SimpleClusterDistances(metric, assignments, points)
end

function update!(dists::SimpleClusterDistances{M},
                 assignments::AbstractVector{<:Integer},
                 points::AbstractMatrix{<:Real}) where M
    @assert length(assignments) == size(points, 2)
    check_assignments(assignments, nclusters(dists))
    append!(dists.assignments, assignments)
    n = size(dists.point_dists, 1)
    length(dists.assignments) == n ||
        error("$(typeof(dists)) does not support batch updates: $(length(assignments)) points given, $n expected")
    @inbounds for cluster in assignments
        dists.cluster_sizes[cluster] += 1
    end

    if M === Nothing
        size(point_dists) == (n, n) ||
            throw(DimensionMismatch("points should be a point-to-point distances matrix of ($n, $n) size, $(size(points)) given"))
        copy!(dists.point_dists, point_dists)
    else
        # metric-based SimpleClusterDistances does not support batched updates
        size(points, 2) == n ||
            throw(DimensionMismatch("points should be a point coordinates matrix with $n columns, $(size(points, 2)) found"))
        pairwise!(dists.metric, dists.point_dists, points, dims=2)
    end

    return dists
end

# this function returns matrix r nclusters x n, such that
# r[i, j] is the sum of distances from all i-th cluster points to the point indices[j]
function sumdistances(dists::SimpleClusterDistances,
                      points::Union{AbstractMatrix, Nothing}, # unused as distances are already in point_dists
                      indices::AbstractVector{<:Integer})
    T = eltype(dists.point_dists)
    n = length(dists.assignments)
    S = typeof((one(T)+one(T))/2)
    r = zeros(S, nclusters(dists), n)
    @inbounds for (jj, j) in enumerate(indices)
        for i = 1:j-1
            r[dists.assignments[i], jj] += dists.point_dists[i,j]
        end
        for i = j+1:n
            r[dists.assignments[i], jj] += dists.point_dists[i,j]
        end
    end
    return r
end

# uses the method from "Distributed Silhouette Algorithm: Evaluating Clustering on Big Data"
# https://arxiv.org/abs/2303.14102
# for SqEuclidean point distances
struct SqEuclideanClusterDistances{T} <: ClusterDistances{T}
    cluster_sizes::Vector{Int} # [nclusters]
    Y::Matrix{T} # [dims, nclusters], the first moments of each cluster (sum of point coords)
    Ψ::Vector{T} # [nclusters], the second moments of each cluster (sum of point coord squares)

    SqEuclideanClusterDistances(::Type{T}, npoints::Union{Integer, Nothing}, dims::Integer,
                                nclusters::Integer) where T<:Real =
        new{T}(zeros(Int, nclusters), zeros(T, dims, nclusters), zeros(T, nclusters))
end

ClusterDistances(::Type{T}, metric::SqEuclidean, npoints::Union{Integer, Nothing},
                 dims::Integer, nclusters::Integer) where T<:Real =
    SqEuclideanClusterDistances(T, npoints, dims, nclusters)

function update!(dists::SqEuclideanClusterDistances,
                 assignments::AbstractVector{<:Integer},
                 points::AbstractMatrix{<:Real})
    # x dims are [D,N]
    d, n = size(points)
    k = length(cluster_sizes(dists))
    check_assignments(assignments, k)
    n == length(assignments) || throw(DimensionMismatch("points count ($n) does not match assignments length $(length(assignments)))"))
    d == size(dists.Y, 1) || throw(DimensionMismatch("points dims ($(size(points, 1))) do no must match ClusterDistances dims ($(size(dists.Y, 1)))"))
    # precompute moments and update counts
    @inbounds for (i, cluster) in enumerate(assignments)
        point = view(points, :, i) # switch to eachslice() once Julia-1.0 support is dropped
        dists.cluster_sizes[cluster] += 1
        dists.Y[:, cluster] .+= point
        dists.Ψ[cluster] += sum(abs2, point)
    end
    return dists
end

# sum distances from each cluster to each point in `points`, [nclusters, n]
function sumdistances(dists::SqEuclideanClusterDistances,
                      points::AbstractMatrix,
                      indices::AbstractVector{<:Integer})
    @assert size(points, 2) == length(indices)
    point_norms = sum(abs2, points; dims=1) # [1,n]
    return dists.cluster_sizes .* point_norms .+
        reshape(dists.Ψ, nclusters(dists), 1) .-
        2 * (transpose(dists.Y) * points)
end
