using LinearAlgebra
using SparseArrays

"""
    equivalent_matrices(x::AbstractMatrix)

Returns a collection of matrixes that are equal to the input `x`, but of a different type.
Useful for testing if things still work on different types of matrix.
"""
function equivalent_matrices(x::AbstractMatrix)
    mats = [
        Base.PermutedDimsArray(x, (1,2)),  # identity permutation
        view(x, :, :),
        view(x, collect.(axes(x))...),  # breaks `strides`
        sparse(x),
    ]
    if issymmetric(x)
        append!(mats, [
            Symmetric(x),
            Transpose(x),
        ])
    end
    return mats
end
