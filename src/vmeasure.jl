# V-measure of contingency table
function _vmeasure(A::AbstractMatrix{<:Integer}; β::Real)
    (β >= 0) || throw(ArgumentError("β should be nonnegative"))

    N = sum(A)
    (N == 0.0) && return 0.0

    entA = entropy(A)
    entArows = entropy(sum(A, dims=2))
    entAcols = entropy(sum(A, dims=1))

    hck = (entA - entAcols)/N
    hkc = (entA - entArows)/N
    hc = entArows/N + log(N)
    hk = entAcols/N + log(N)

    # Homogeneity
    h = hc == 0.0 ? 1.0 : 1.0 - hck/hc
    # Completeness
    c = hk == 0.0 ? 1.0 : 1.0 - hkc/hk

    # V-measure
    V_β = (1 + β)*h*c/(β*h + c)
    return V_β
end

"""
    vmeasure(a, b; [β = 1.0]) -> Float64

V-measure between the two clusterings.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

The `β` parameter defines trade-off between _homogeneity_ and _completeness_:
 * if ``β > 1``, _completeness_ is weighted more strongly,
 * if ``β < 1``, _homogeneity_ is weighted more strongly.

# References
> Andrew Rosenberg and Julia Hirschberg, 2007. *V-Measure: A conditional
> entropy-based external cluster evaluation measure*
"""
vmeasure(a, b; β::Real = 1.0) = _vmeasure(counts(a, b), β=float(β))
