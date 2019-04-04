# Common utilities

##### common types

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
