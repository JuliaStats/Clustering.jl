"""
    confusion(a::Union{ClusteringResult, AbstractVector},
              b::Union{ClusteringResult, AbstractVector}) -> Matrix{Int}

Return 2×2 confusion matrix `C` that represents partition co-occurrence or
similarity matrix between two clusterings by considering all pairs of samples
and counting  pairs that are assigned into the same or into different clusters
under the true and predicted clusterings.

Considering a pair of samples that is in the same group as a **positive pair**,
and a pair is in the different group as a **negative pair**, then the count of
true positives is `C₁₁`, false negatives is `C₁₂`, false positives `C₂₁`, and
true negatives is `C₂₂`:

|  | Positive | Negative |
|:--:|:-:|:-:|
|Positive|C₁₁|C₁₂|
|Negative|C₂₁|C₂₂|
"""
function confusion(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    c = counts(a, b)

    n = sum(c)
    nis = sum(abs2, sum(c, dims=2))        # sum of squares of sums of rows
    njs = sum(abs2, sum(c, dims=1))        # sum of squares of sums of columns

    t2 = sum(abs2, c)                      # sum over rows & columns of nij^2
    t3 = nis + njs
    C = [(t2 - n)÷2 (nis - t2)÷2; (njs - t2)÷2 (t2 + n^2 - t3)÷2]
    return C
end

confusion(a::ClusteringResult, b::ClusteringResult) =
    confusion(assignments(a), assignments(b))
confusion(a::AbstractVector{<:Integer}, b::ClusteringResult) =
    confusion(a, assignments(b))
confusion(a::ClusteringResult, b::AbstractVector{<:Integer}) =
    confusion(assignments(a), b)

