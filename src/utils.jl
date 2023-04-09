# Common utilities
##### common types
using Distances

"""
    ClusteringResult

Base type for the output of clustering algorithm.
"""
abstract type ClusteringResult end

# generic functions

"""
    nclusters(R::ClusteringResult) -> Int

Get the number of clusters.
"""
nclusters(R::ClusteringResult) = length(R.counts)

"""
    counts(R::ClusteringResult) -> Vector{Int}

Get the vector of cluster sizes.

`counts(R)[k]` is the number of points assigned to the ``k``-th cluster.
"""
counts(R::ClusteringResult) = R.counts

"""
    wcounts(R::ClusteringResult) -> Vector{Float64}
    wcounts(R::FuzzyCMeansResult) -> Vector{Float64}

Get the weighted cluster sizes as the sum of weights of points assigned to each
cluster.

For non-weighted clusterings assumes the weight of every data point is 1.0,
so the result is equivalent to `convert(Vector{Float64}, counts(R))`.
"""
wcounts(R::ClusteringResult) = convert(Vector{Float64}, counts(R))

"""
    assignments(R::ClusteringResult) -> Vector{Int}

Get the vector of cluster indices for each point.

`assignments(R)[i]` is the index of the cluster to which the ``i``-th point
is assigned.
"""
assignments(R::ClusteringResult) = R.assignments


##### convert display symbol to disp level
const DisplayLevels = Dict(:none => 0, :final => 1, :iter => 2)

display_level(s::Symbol) = get(DisplayLevels, s) do
    throw(ArgumentError("Invalid value for the 'display' option: $s."))
end

##### update minimum value

function updatemin!(r::AbstractArray, x::AbstractArray)
    n = length(r)
    length(x) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    @inbounds for i = 1:n
        xi = x[i]
        if xi < r[i]
            r[i] = xi
        end
    end
    return r
end


"""
    assign_clusters(X::AbstractMatrix{<:Real}, R::ClusteringResult; ...) -> Vector{Int}

Assign the samples specified as the columns of `X` to the corresponding clusters from `R`.

# Arguments
- `X`: Input data to be clustered.
- `R`: Fitted clustering result.
- `distance`: SemiMertric used to compute distances between vectors and clusters centroids.
- `pairwise_computation`: Boolean specifying whether to compute and store pairwise distances.

"""
function assign_clusters(
    X::AbstractMatrix{T}, 
    R::ClusteringResult;
    distance::SemiMetric = SqEuclidean(),
    pairwise_computation::Bool = true) where {T} 

    if !(typeof(R) <: KmeansResult)
        throw(MethodError(assign_clusters,
              "NotImplemented: assign_clusters not implemented for R of type $(typeof(R))"))
    end

end