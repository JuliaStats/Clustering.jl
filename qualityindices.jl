


function _check_qualityindex_argument(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:AbstractFloat},
        assignments::AbstractVector{<:Integer},
    )
    d, n = size(X)
    dc, k = size(centers)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (1 <= k <= n) || throw(ArgumentError("Cluster number k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    for j = 1:n
        (1 <= assignments[j] <= k) || throw(ArgumentError("Bad assignments[$j]=$(assignments[j]): should be in 1:$k range."))
    end
end

function _check_qualityindex_argument(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:AbstractFloat},
        weights::AbstractMatrix{<:AbstractFloat},
        fuzziness::Real,
    )
    d, n = size(X)
    dc, k = size(centers)
    nw, kw = size(weights)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    n == nw || throw(DimensionMismatch("Inconsistent data length for `X` and `weights`."))
    k == kw || throw(DimensionMismatch("Inconsistent number of clusters for `centers` and `weights`."))
    (1 <= k <= n) || throw(ArgumentError("Cluster number k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    all(>=(0), weights) || throw(ArgumentError("All weights must be larger or equal 0."))
    1 < fuzziness || throw(ArgumentError("Fuzziness must be greater than 1 ($fuzziness given)"))
end

function _check_qualityindex_argument(assignments::AbstractVector{<:Integer}, dist::AbstractMatrix{<:Real})
    n, m = size(dist)
    na = length(assignments)
    n == m || throw(ArgumentError("Distance matrix must be square."))
    n == na || throw(DimensionMismatch("Inconsistent array dimensions for distance matrix and assignments."))
end