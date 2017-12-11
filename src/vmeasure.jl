# V-measure of contingency table
function _vmeasure(A::Matrix{Int}; β = 1.0)
    @assert β >= 0 "β should be nonnegative"

    C, K = size(A)
    N = sum(A)

    # Homogeneity
    hck = 0.0
    for k in 1:K
        Ak = view(A, :, k)
        d = sum(Ak)
        for c in 1:C
            if Ak[c] != 0 && d != 0
                hck += log(Ak[c]/d) * Ak[c]/N
            end
        end
    end
    hck = -hck

    hc = entropy(sum(A,2)./N)

    h = (hc == 0.0 || hck == 0.0) ? 1.0 : 1.0 - hck/hc

    # Completeness
    hkc = 0.0
    for c in 1:C
        Ac = view(A, c, :)
        d = sum(Ac)
        for k in 1:K
            if Ac[k] != 0 && d != 0
                hkc += log(Ac[k]/d) * Ac[k]/N
            end
        end
    end
    hkc = -hkc

    hk = entropy(sum(A,1)./N)

    c = (hk == 0.0 || hkc == 0.0) ? 1.0 : 1.0 - hkc/hk

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
