# Common utilities

##### common types

"""
Base type for the output of clustering algorithm.
"""
abstract type ClusteringResult end

# generic functions

"""
    nclusters(R::ClusteringResult)

Get the number of clusters.
"""
nclusters(R::ClusteringResult) = length(R.counts)

"""
    counts(R::ClusteringResult)

Get the vector of cluster sizes.

`counts(R)[k]` is the number of points assigned to the ``k``-th cluster.
"""
counts(R::ClusteringResult) = R.counts

"""
    assignments(R::ClusteringResult)

Get the vector of cluster indices for each point.

`assignments(R)[i]` is the index of the cluster to which the ``i``-th point
is assigned.
"""
assignments(R::ClusteringResult) = R.assignments


##### convert display symbol to disp level

display_level(s::Symbol) =
    s == :none ? 0 :
    s == :final ? 1 :
    s == :iter ? 2 :
    error("Invalid value for the option 'display'.")


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
