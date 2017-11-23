"""V-measure of contingency table"""
function _vmeasure(A::Matrix{Int}; β = 1.0)
    C, K = size(A)
    N = sum(A)

    # Homogeneity
    hck = 0.0
    for k in 1:K
        d = sum(A[:,k])
        for c in 1:C
            if A[c,k] != 0 && d != 0
                hck += log(A[c,k]/d) * A[c,k]/N
            end
        end
    end
    hck = -hck

    hc = 0.0
    for c in 1:C
        n = sum(A[c,:]) / N
        if n != 0.0
            hc += log(n) * n
        end
    end
    hc = -hc

    h = (hc == 0.0 || hck == 0.0) ? 1 : 1 - hck/hc

    # Completeness
    hkc = 0.0
    for c in 1:C
        d = sum(A[c,:])
        for k in 1:K
            if A[c,k] != 0 && d != 0
                hkc += log(A[c,k]/d) * A[c,k]/N
            end
        end
    end
    hkc = -hkc

    hk = 0.0
    for k in 1:K
        n = sum(A[:,k]) / N
        if n != 0.0
            hk += log(n) * n
        end
    end
    hk = -hk

    c = (hk == 0.0 || hkc == 0.0) ? 1 : 1 - hkc/hk

    # V-measure
    V_β = (1 + β)*h*c/(β*h + c)
    return V_β
end

"""V-measure between two clustering assignments.

`β` parameter defines trade-off between homogeneity and completeness,
if β is greater than 1 completeness is V-measure  weighted more strongly in the completeness,
if β is less than 1, homogeneity is weighted more strongly.

Andrew Rosenberg and Julia Hirschberg, 2007. "V-Measure: A conditional entropy-based external cluster evaluation measure"
"""
vmeasure(assign1::Vector{Int}, assign2::Vector{Int}; β = 1.0) =
    _vmeasure(counts(assign1,assign2,(1:maximum(assign1),1:maximum(assign2))), β = β)
vmeasure(R1::ClusteringResult, assign::Vector{Int}; β = 1.0) =
    vmeasure(assignments(R), assign, β = β)
vmeasure(R1::ClusteringResult, R2::ClusteringResult; β = 1.0) =
    vmeasure(assignments(R1), assignments(R2), β = β)
