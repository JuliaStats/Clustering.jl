"""
    confusion([T = Int],
              a::Union{ClusteringResult, AbstractVector},
              b::Union{ClusteringResult, AbstractVector}) -> Matrix{T}

Calculate the *confusion matrix* of the two clusterings.

Returns the 2×2 confusion matrix `C` of type `T` (`Int` by default) that
represents partition co-occurrence or similarity matrix between two clusterings
`a` and `b` by considering all pairs of samples and counting pairs that are
assigned into the same or into different clusters.

Considering a pair of samples that is in the same group as a **positive pair**,
and a pair is in the different group as a **negative pair**, then the count of
true positives is `C₁₁`, false negatives is `C₁₂`, false positives `C₂₁`, and
true negatives is `C₂₂`:

|  | Positive | Negative |
|:--:|:-:|:-:|
|Positive|C₁₁|C₁₂|
|Negative|C₂₁|C₂₂|
"""
function confusion(::Type{T}, a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) where T<:Union{Integer, AbstractFloat}
    cc = counts(a, b)
    c = eltype(cc) === T ? cc : convert(Matrix{T}, cc)

    n = sum(c)
    nis = sum(abs2, sum!(zeros(T, (size(c, 1), 1)), c))
    (nis < 0) && OverflowError("sum of squares of sums of rows overflowed")
    njs = sum(abs2, sum!(zeros(T, (1, size(c, 2))), c))
    (njs < 0) && OverflowError("sum of squares of sums of columns overflowed")

    t2 = sum(abs2, c)
    (t2 < 0) && OverflowError("sum of squares of matrix elements overflowed")
    t3 = nis + njs
    C = [(t2 - n)÷2 (nis - t2)÷2; (njs - t2)÷2 (t2 + n^2 - t3)÷2]
    return C
end

confusion(T, a::ClusteringResultOrAssignments,
             b::ClusteringResultOrAssignments) =
    confusion(T, assignments(a), assignments(b))

confusion(a::ClusteringResultOrAssignments,
          b::ClusteringResultOrAssignments) =
    confusion(Int, a, b)
