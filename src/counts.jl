# wrapper for StatsBase.counts(a::Vector, b::Vector, (1:maxA, 1:maxB))
function _counts(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    n = length(a)
    n == length(b) ||
        throw(DimensionMismatch("Assignment vectors have different lengths ($n and $(length(b)))"))
    # NOTE: StatsBase.counts() throws ArgumentError for empty vectors
    (n == 0) && return Matrix{Int}(undef, 0, 0)
    minA, maxA = extrema(a)
    minB, maxB = extrema(b)
    (minA > 0 && minB > 0) ||
        throw(ArgumentError("Cluster indices should be positive integers"))
    # note: ignoring minA/minB, always start from 1 to match
    #       cluster indices and counts matrix positions
    return counts(a, b, (1:maxA, 1:maxB))
end

"""
    counts(a::ClusteringResult, b::ClusteringResult) -> Matrix{Int}
    counts(a::ClusteringResult, b::AbstractVector{<:Integer}) -> Matrix{Int}
    counts(a::AbstractVector{<:Integer}, b::ClusteringResult) -> Matrix{Int}

Calculate the *cross tabulation* (aka *contingency matrix*) for the two
clusterings of the same data points.

Returns the ``n_a × n_b`` matrix `C`, where ``n_a`` and ``n_b`` are the
numbers of clusters in `a` and `b`, respectively, and `C[i, j]` is
the size of the intersection of `i`-th cluster from `a` and `j`-th cluster
from `b`.

The clusterings could be specified either as [`ClusteringResult`](@ref)
instances or as vectors of data point assignments.
"""
counts(a::ClusteringResult, b::ClusteringResult) =
    _counts(assignments(a), assignments(b))
counts(a::AbstractVector{<:Integer}, b::ClusteringResult) =
    _counts(a, assignments(b))
counts(a::ClusteringResult, b::AbstractVector{<:Integer}) =
    _counts(assignments(a), b)
