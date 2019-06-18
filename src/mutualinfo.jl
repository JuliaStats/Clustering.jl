# Mutual Information

function _mutualinfo(A::AbstractMatrix{<:Integer})
    N = sum(A)
    (N == 0.0) && return 0.0

    aᵢ = sum(A, dims=2)
    bⱼ = sum(A, dims=1)
    entA = entropy(A)
    entArows = entropy(aᵢ)
    entAcols = entropy(bⱼ)

    hck = (entA - entAcols)/N
    hc = entArows/N + log(N)
    hk = entAcols/N + log(N)

    mi = hc - hck
    nmi = 2*mi/(hc+hk)

    return mi, nmi
end

"""
    mutualinfo(a, b) -> Float64

Compute the *mutual information* between the two clusterings of the same
data points.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

Returns a tuple of indices:
  - mutual information
  - normalized mutual information (sum)

# References
> Vinh, Epps, and Bailey, (2009). “Information theoretic measures for clusterings comparison”.
Proceedings of the 26th Annual International Conference on Machine Learning - ICML ‘09.
"""
mutualinfo(a, b) = _mutualinfo(counts(a, b))
