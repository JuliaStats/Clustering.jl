# Mutual Information

function _mutualinfo(A::AbstractMatrix{<:Integer}, normed::Bool)
    N = sum(A)
    (N == 0.0) && return 0.0

    rows = sum(A, dims=2)
    cols = sum(A, dims=1)
    entA = entropy(A)
    entArows = entropy(rows)
    entAcols = entropy(cols)

    hck = (entA - entAcols)/N
    hc = entArows/N + log(N)
    hk = entAcols/N + log(N)

    mi = hc - hck
    return if normed
        2*mi/(hc+hk)
    else
        mi
    end
end

"""
    mutualinfo(a, b; normed=true) -> Float64

Compute the *mutual information* between the two clusterings of the same
data points.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

If `normed` parameter is `true` the return value is the normalized mutual information (symmetric uncertainty),
see "Data Mining Practical Machine Tools and Techniques", Witten & Frank 2005.

# References
> Vinh, Epps, and Bailey, (2009). “Information theoretic measures for clusterings comparison”.
Proceedings of the 26th Annual International Conference on Machine Learning - ICML ‘09.
"""
mutualinfo(a, b; normed::Bool=true) = _mutualinfo(counts(a, b), normed)
