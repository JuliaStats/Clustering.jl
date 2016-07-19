function randindex(c1,c2)
    # rand_index - calculates Rand Indices to compare two partitions
    # (AR, RI, MI, HI) = rand(c1,c2), where c1,c2 are vectors listing the
    # class membership, returns the "Hubert & Arabie adjusted Rand index".
    # (AR, RI, MI, HI) = rand(c1,c2) returns the adjusted Rand index,
    # the unadjusted Rand index, "Mirkin's" index and "Hubert's" index.
    #
    # See L. Hubert and P. Arabie (1985) "Comparing Partitions" Journal of
    # Classification 2:193-218

    c = counts(c1,c2,(1:maximum(c1),1:maximum(c2))) # form contingency matrix

    n = round(Int,sum(c))
    nis = sum(sum(c,2).^2)        # sum of squares of sums of rows
    njs = sum(sum(c,1).^2)        # sum of squares of sums of columns

    t1 = binomial(n,2)            # total number of pairs of entities
    t2 = sum(c.^2)                # sum over rows & columnns of nij^2
    t3 = .5*(nis+njs)

    # Expected index (for adjustment)
    nc = (n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))

    A = t1+t2-t3;        # no. agreements
    D = -t2+t3;          # no. disagreements

    if t1 == nc
        # avoid division by zero; if k=1, define Rand = 0
        ARI = 0
    else
        # adjusted Rand - Hubert & Arabie 1985
        ARI = (A-nc)/(t1-nc)
    end

    RI = A/t1            # Rand 1971      # Probability of agreement
    MI = D/t1            # Mirkin 1970    # p(disagreement)
    HI = (A-D)/t1        # Hubert 1977    # p(agree)-p(disagree)

    return (ARI, RI, MI, HI)
end

randindex(R::ClusteringResult, c0::AbstractVector{Int}) = randindex(assignments(R), c0)
randindex(R1::ClusteringResult, R2::ClusteringResult) = randindex(assignments(R2), assignments(R1))
