# Variation of Information

"""
    varinfo(a, b) -> Float64

Compute the *variation of information* between the two clusterings of the same
data points.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

# References
> Meila, Marina (2003). *Comparing Clusterings by the Variation of
> Information.* Learning Theory and Kernel Machines: 173–187.
"""
function varinfo(a, b)
    C = counts(a, b)
    isempty(C) && return 0.0
    countsA = a isa ClusteringResult ? counts(a) : sum(C, dims=2)
    countsB = b isa ClusteringResult ? counts(b) : sum(C, dims=1)
    I = 0.0
    @inbounds for (i, ci) in enumerate(countsA), (j, cj) in enumerate(countsB)
        cij = C[i, j]
        if cij > 0.0
            I += cij * log(cij*cij / (ci*cj))
        end
    end
    return -I/sum(countsA)
end
