# V-measure of contingency table
function _vmeasure(A::AbstractMatrix{Int}; β::Real)
    @assert β >= 0 "β should be nonnegative"

    N = sum(A)
    (N == 0.0) && return 0.0

    entA = entropy(A)
    entArows = entropy(sum(A, 2))
    entAcols = entropy(sum(A, 1))

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
    vmeasure(assign1, assign2; β = 1.0 )

V-measure between two clustering assignments.

`assign1` and `assign2` can be either `ClusteringResult` objects or assignments vectors, `AbstractVector{Int}`.

`β` parameter defines trade-off between homogeneity and completeness,
if `β` is greater than 1, completeness is weighted more strongly in the completeness,
if `β` is less than 1, homogeneity is weighted more strongly.

*Ref:* Andrew Rosenberg and Julia Hirschberg, 2007. "V-Measure: A conditional entropy-based external cluster evaluation measure"
"""
function vmeasure(clusters1::Union{AbstractVector{Int}, ClusteringResult},
                  clusters2::Union{AbstractVector{Int}, ClusteringResult}; β::Real = 1.0)
    assign1 = isa(clusters1, AbstractVector) ? clusters1 : assignments(clusters1)
    assign2 = isa(clusters2, AbstractVector) ? clusters2 : assignments(clusters2)
    _vmeasure(counts(assign1,assign2,(1:maximum(assign1),1:maximum(assign2))), β = β)
end
