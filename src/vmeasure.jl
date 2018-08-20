# V-measure of contingency table
function _vmeasure(A::AbstractMatrix{Int}, β::Real)
    β < 0 && throw(ArgumentError("β should be nonnegative"))

    C, K = size(A)
    N = sum(A)
    N == 0.0 && return 1.0

    entA = entropy(A)
    entArows = entropy(sum(A, dims=2))
    entAcols = entropy(sum(A, dims=1))

    # Homogeneity
    hck = (entA - entAcols)/N
    hc = entArows/N + log(N)
    h = hc == 0.0 ? 1.0 : 1.0 - hck/hc

    # Completeness
    hkc = (entA - entArows)/N
    hk = entAcols/N + log(N)
    c = hk == 0.0 ? 1.0 : 1.0 - hkc/hk

    # V-measure
    V_β = (1 + β)*h*c/(β*h + c)
    return V_β
end

"""
    vmeasure(assign1, assign2; β = 1.0)

Compute V-measure value between two clustering assignments.

`assign1` the vector of assignments for the first clustering, `assign2` the vector of assignments for the second clustering.
`assign1` and `assign2` can be either `ClusteringResult` objects or assignments vectors, `AbstractVector{Int}`.

`β` parameter defines trade-off between homogeneity and completeness,
if `β` > 1, the completeness is weighted more strongly in the V-measure, if `β` < 1, the homogeneity is weighted more strongly.

*Ref:* Andrew Rosenberg and Julia Hirschberg, 2007. "V-Measure: A conditional entropy-based external cluster evaluation measure"
"""
function vmeasure(assign1::Union{AbstractVector{Int}, ClusteringResult},
                  assign2::Union{AbstractVector{Int}, ClusteringResult}; β::Real = 1.0)
    _assign1 = isa(assign1, AbstractVector) ? assign1 : assignments(assign1)
    _assign2 = isa(assign2, AbstractVector) ? assign2 : assignments(assign2)
    return _vmeasure(counts(_assign1, _assign2,
                     (1:maximum(_assign1), 1:maximum(_assign2))), β)
end
