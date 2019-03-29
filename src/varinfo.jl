# Variation of Information

"""
    varinfo(k1::Int, a1::AbstractVector{Int}, k2::Int, a2::AbstractVector{Int})
    varinfo(R::ClusteringResult, k0::Int, a0::AbstractVector{Int})
    varinfo(R1::ClusteringResult, R2::ClusteringResult)

Compute the variation of information between the two clusterings.

Each clustering is provided either as an instance of [`ClusteringResult`](@ref)
subtype or as a pair of arguments:
 - a number of clusters (`k1`, `k2`, `k0`)
 - a vector of point to cluster assignments (`a1`, `a2`, `a0`).
"""
function varinfo(k1::Int, a1::AbstractVector{Int},
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

    return H1 + H2 - I * 2.0
end

varinfo(R::ClusteringResult, k0::Int, a0::AbstractVector{Int}) =
    varinfo(nclusters(R), assignments(R), k0, a0)

varinfo(R1::ClusteringResult, R2::ClusteringResult) =
    varinfo(nclusters(R1), assignments(R1),
            nclusters(R2), assignments(R2))
