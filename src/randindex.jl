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
> Journal of Classification 2 (1): 193-218

> Meila, Marina (2003). *Comparing Clusterings by the Variation of
> Information.* Learning Theory and Kernel Machines: 173-187.

> Steinley, Douglas (2004). *Properties of the Hubert-Arabie Adjusted
> Rand Index.* Psychological Methods, Vol. 9, No. 3: 386-396
"""
function randindex(a, b)
    c11, c21, c12, c22 = confusion(Float64, a, b) # Table 2 from Steinley 2004

    t = c11 + c12 + c21 + c22   # total number of pairs of entities
    A = c11 + c22
    D = c12 + c21

    # expected index
    ERI = (c11+c12)*(c11+c21)+(c21+c22)*(c12+c22)
    # adjusted Rand - Hubert & Arabie 1985
    ARI = D == 0 ? 1.0 : (t*A - ERI)/(abs2(t) - ERI) # (9) from Steinley 2004

    RI = A/t            # Rand 1971      # Probability of agreement
    MI = D/t            # Mirkin 1970    # p(disagreement)
    HI = (A-D)/t        # Hubert 1977    # p(agree)-p(disagree)

    return (ARI, RI, MI, HI)
end
