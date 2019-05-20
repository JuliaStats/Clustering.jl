"""
    randindex(a, b) -> NTuple{4, Float64}

Compute the tuple of Rand-related indices between the clusterings `c1` and `c2`.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

Returns a tuple of indices:
  - Hubert & Arabie Adjusted Rand index
  - Rand index (agreement probability)
  - Mirkin's index (disagreement probability)
  - Hubert's index (``P(\\mathrm{agree}) - P(\\mathrm{disagree})``)

# References
> Lawrence Hubert and Phipps Arabie (1985). *Comparing partitions.*
> Journal of Classification 2 (1): 193–218

> Meila, Marina (2003). *Comparing Clusterings by the Variation of
> Information.* Learning Theory and Kernel Machines: 173–187.
"""
function randindex(a, b)
    c = counts(a, b)

    n = sum(c)
    nis = sum(abs2, sum(c, dims=2))        # sum of squares of sums of rows
    njs = sum(abs2, sum(c, dims=1))        # sum of squares of sums of columns

    t1 = binomial(n, 2)                    # total number of pairs of entities
    t2 = sum(abs2, c)                      # sum over rows & columnns of nij^2
    t3 = .5*(nis+njs)

    # Expected index (for adjustment)
    nc = (n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))

    A = t1+t2-t3;        # agreements count
    D = -t2+t3;          # disagreements count

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
