# simple program to test affinity propagation

using Test
using Distances
using Clustering
using LinearAlgebra
using Random, StableRNGs
using Statistics
include("test_helpers.jl")

@testset "affinityprop() (affinity propagation)" begin

    @testset "Argument checks" begin
        @test_throws ArgumentError affinityprop(randn(2, 3))
        @test_throws ArgumentError affinityprop(randn(1, 1))
        @test_throws ArgumentError affinityprop(randn(2, 2), tol=0.0)
        @test_throws ArgumentError affinityprop(randn(2, 2), damp=-0.1)
        @test_throws ArgumentError affinityprop(randn(2, 2), damp=1.0)
        @test affinityprop(randn(2, 2), damp=0.5, tol=0.5) isa AffinityPropResult
        for disp in keys(Clustering.DisplayLevels)
            @test affinityprop(randn(2, 2), tol=0.1, display=disp) isa AffinityPropResult
        end
    end

    rng = StableRNG(34568)
    d = 10
    n = 500
    x = rand(rng, d, n)
    S = -pairwise(Euclidean(), x, x, dims=2)

    # set diagonal value to median value
    S = S - diagm(0 => diag(S)) + median(S)*I

    R = affinityprop(S)

    @test isa(R, AffinityPropResult)
    k = length(R.exemplars)
    @test 0 < k < n
    @test length(R.assignments) == n
    @test all(R.assignments .>= 1) && all(R.assignments .<= k)
    @test all(R.assignments[R.exemplars] .== collect(1:k))

    @test length(R.counts) == k
    @test sum(R.counts) == n
    for i = 1:k
        @test R.counts[i] == count(==(i), R.assignments)
    end

    @testset "Support for arrays other than Matrix{T}" begin
        @testset "$(typeof(M))" for M in equivalent_matrices(S)
            R2 = affinityprop(M)
            @test R2.assignments == R.assignments
        end
    end

    #= compare with python result
    the reference assignments were computed using python sklearn:
    ```julia
    using PyCall

    @pyimport sklearn.cluster as cl
    af = cl.AffinityPropagation(affinity="precomputed")[:fit]( S )

    ref_assignments = af[:labels_] .+ 1
    ```
    =#
    ref_assignments = [7, 30, 53, 30, 43, 55, 19, 31, 23, 16, 31, 31, 1, 14, 47,
                       45, 54, 48, 8, 55, 39, 45, 14, 47, 24, 27, 28, 20, 9, 8,
                       3, 32, 17, 16, 54, 50, 2, 46, 32, 30, 21, 52, 19, 55, 2,
                       47, 49, 3, 2, 45, 43, 27,  51, 4, 16, 46, 55, 11, 35, 1,
                       56, 40, 45, 33, 26, 26, 51, 13, 18,  4, 55, 19, 3, 52, 39,
                       5, 6, 43, 21, 16, 20, 34, 16, 9, 19, 3, 30, 48, 43, 30, 1,
                       17, 26, 30, 6, 27, 18, 2, 40, 3, 53, 7, 37, 7, 4, 21, 14,
                       49, 4, 39, 29, 34, 23, 22, 41, 44, 48, 39, 7, 2, 1, 23,
                       41, 8, 53, 31, 37, 54, 28, 2, 17, 9, 1, 10, 11, 34, 14,
                       8, 39, 55, 43, 37, 24, 15, 53, 4, 44, 40, 12, 51, 42, 50,
                       13, 15, 5, 34, 27, 2, 12, 14, 48, 10, 49, 4, 36, 53, 36,
                       24, 22, 36, 45, 22, 52, 19, 31, 22, 46, 10, 25, 42, 15,
                       25, 53, 16, 5, 25, 14, 51, 19, 50, 32, 54, 4, 45, 17, 56,
                       18, 41, 23, 39, 7, 53, 56, 30, 37, 12, 16, 19, 20, 20, 42,
                       39, 16, 45, 37, 17, 52, 15, 21, 6, 33, 41, 1, 34, 22, 19,
                       54, 16, 44, 31, 23, 11, 7, 24, 11, 53, 49, 55, 46, 43, 25,
                       51, 55, 25, 47, 16, 46, 26, 55, 14, 53, 3, 44, 34, 26, 19,
                       49, 35, 3, 34, 32, 27, 42, 28, 42, 42, 54, 2, 29, 21, 20,
                       25, 19, 9, 50, 3, 30, 14, 32, 43, 34, 12, 5, 6, 3, 50, 27,
                       50, 52, 51, 46, 39, 14, 12, 30, 32, 19, 19, 43, 19, 25,
                       40, 31, 25, 52, 30, 37, 27, 20, 8, 22, 39, 55, 25, 21, 31,
                       17, 16, 15, 31, 29, 17, 5, 32, 38, 4, 16, 52, 48, 18, 17,
                       41, 4, 23, 3, 29, 44, 50, 40, 52, 29, 9, 36, 15, 33, 13,
                       52, 20, 14, 38, 30, 24, 34, 40, 41, 21, 22, 24, 20, 15, 35,
                       36, 47, 45, 45, 23, 37, 38, 19, 26, 16, 39, 16, 31, 28, 27,
                       40, 41, 30, 17, 3, 14, 52, 31, 38, 28, 37, 34, 44, 53, 32,
                       14, 5, 23, 42, 43, 44, 22, 55, 12, 39, 20, 37, 28, 19, 33,
                       54, 33, 1, 10, 45, 6, 46, 47, 50, 29, 38, 26, 48, 20, 49,
                       32, 6, 22, 39, 34, 27, 25, 53, 28, 50, 41, 43, 49, 3, 51,
                       10, 27, 51, 28, 23, 44, 24, 20, 4, 28, 29, 11, 33, 52, 19,
                       4, 9, 14, 36, 34, 13, 31, 27, 41, 47, 35, 37, 53, 6, 56,
                       53, 3, 39, 7, 3, 29, 26, 1, 32, 3, 24, 38, 14, 6, 54, 23,
                       27, 17, 56, 29, 35, 46, 31, 46, 55, 56, 20, 32, 54, 46,
                       48, 26, 1, 48]
    @test randindex(R.assignments, ref_assignments)[2] == 1
end
