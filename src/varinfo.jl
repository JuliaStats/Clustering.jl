# Variation of Information
const _varinfo_default_variant = :Djoint

function information(k1::Int, a1::AbstractVector{Int},
                 k2::Int, a2::AbstractVector{Int})

    # check input arguments
    n = length(a1)
    length(a2) == n || throw(DimensionMismatch("Inconsistent array length."))

    # count & compute probabilities
    p1 = zeros(k1)
    p2 = zeros(k2)
    P = zeros(k1, k2)

    for i = 1:n
        @inbounds l1 = a1[i]
        @inbounds l2 = a2[i]
        p1[l1] += 1.0
        p2[l2] += 1.0
        P[l1, l2] += 1.0
    end

    for i = 1:k1
        @inbounds p1[i] /= n
    end
    for i = 1:k2
        @inbounds p2[i] /= n
    end
    for i = 1:(k1*k2)
        @inbounds P[i] /= n
    end

    # compute variation of information

    H1 = entropy(p1)
    H2 = entropy(p2)

    I = 0.0
    for j = 1:k2, i = 1:k1
        pi = p1[i]
        pj = p2[j]
        pij = P[i,j]
        if pij > 0.0
            I += pij * log(pij / (pi * pj))
        end
    end

    return I, H1, H2
end

function varinfo(k1::Int, a1::AbstractVector{Int},
                 k2::Int, a2::AbstractVector{Int},
                 variant::Symbol=_varinfo_default_variant)
    I, H1, H2 = information(k1, a1, k2, a2)
    if variant == :Djoint
        v = H1 + H2 - I * 2.0
    elseif variant == :Dmax
        v = max(H1,H2) - I 
    elseif variant == :djoint
        v = (1 - 2*I/(H1+H2))
    elseif variant == :dmax
        v = 1 - (I/max(H1, H2))
    end
    return v
end

varinfo(R::ClusteringResult, k0::Int, a0::AbstractVector{Int}, variant::Symbol=_varinfo_default_variant) = 
    varinfo(nclusters(R), assignments(R), k0, a0, variant)

varinfo(R1::ClusteringResult, R2::ClusteringResult, variant::Symbol=_varinfo_default_variant) = 
    varinfo(nclusters(R1), assignments(R1), 
            nclusters(R2), assignments(R2),
            variant)

