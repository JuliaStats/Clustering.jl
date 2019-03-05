# simple program to test affinity propagation

using Test
using Distances
using Clustering
using LinearAlgebra
using Random
using Statistics 

@testset "affinityprop() (affinity propagation)" begin

    Random.seed!(34568)

    d = 10
    n = 500
    x = rand(d, n)
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
    
    #= compare with python result
    the reference assignments were computed using python sklearn:
    ```julia
    using PyCall

    @pyimport sklearn.cluster as cl
    af = cl.AffinityPropagation(affinity="precomputed")[:fit]( S )

    ref_assignments = af[:labels_] .+ 1
    ```
    =#
    ref_assignments = [11, 47, 23, 1, 3, 8, 54, 2, 6, 29, 11, 43, 2, 20, 13, 40, 
                       3, 48, 3, 32, 19, 59, 4, 28, 59, 24, 5, 36, 6, 23, 49, 19, 
                       42, 14, 40, 29, 20, 49, 53, 48, 51, 55, 57, 32, 32, 10, 48, 
                       12, 13, 7, 1, 57, 52, 7, 3, 47, 43, 10, 53, 40, 52, 8, 42, 
                       15, 23, 47, 32, 9, 10, 42, 52, 43, 49, 32, 49, 12, 5, 53, 
                       54, 58, 14, 21, 7, 16, 14, 17, 12, 15, 38, 15, 33, 11, 27, 
                       42, 56, 14, 42, 41, 34, 26, 55, 46, 47, 9, 25, 23, 19, 47, 
                       34, 38, 53, 10, 9, 9, 43, 59, 43, 51, 12, 39, 29, 12, 13, 
                       9, 52, 1, 13, 36, 11, 21, 58, 38, 2, 31, 18, 19, 16, 14, 6, 
                       50, 15, 16, 21, 26, 17, 21, 43, 25, 18, 39, 57, 19, 41, 4, 
                       31, 20, 15, 16, 34, 10, 28, 45, 21, 13, 13, 21, 43, 2, 46, 
                       52, 12, 26, 21, 14, 52, 21, 36, 6, 22, 25, 45, 25, 12, 1, 3, 
                       25, 31, 55, 28, 32, 9, 32, 44, 33, 6, 17, 2, 58, 32, 38, 20, 
                       3, 12, 51, 23, 51, 1, 32, 12, 35, 13, 39, 48, 40, 7, 57, 4, 
                       38, 51, 57, 24, 20, 30, 25, 28, 55, 32, 2, 28, 26, 27, 23, 
                       46, 46, 9, 12, 49, 52, 17, 24, 44, 23, 54, 46, 9, 28, 32, 
                       20, 27, 22, 57, 60, 61, 59, 56, 41, 8, 26, 20, 12, 36, 26, 
                       23, 48, 17, 29, 19, 41, 20, 4, 29, 55, 43, 11, 24, 52, 42, 
                       30, 37, 32, 11, 59, 6, 53, 59, 33, 52, 11, 31, 3, 52, 36, 
                       34, 40, 23, 48, 50, 16, 32, 52, 55, 47, 56, 25, 13, 12, 33, 
                       35, 20, 57, 61, 47, 40, 29, 31, 34, 37, 19, 2, 59, 57, 35, 
                       44, 50, 30, 36, 27, 25, 50, 5, 38, 61, 8, 60, 23, 60, 37, 
                       9, 6, 9, 25, 33, 23, 27, 55, 25, 7, 38, 55, 12, 44, 29, 7, 
                       14, 34, 56, 39, 43, 14, 27, 17, 29, 51, 38, 40, 41, 42, 
                       16, 42, 14, 19, 44, 43, 34, 60, 44, 20, 3, 45, 61, 57, 33, 
                       46, 44, 40, 22, 8, 12, 43, 46, 20, 26, 43, 7, 61, 47, 59, 
                       38, 48, 1, 49, 48, 35, 22, 50, 44, 43, 1, 53, 50, 52, 53, 
                       41, 49, 6, 51, 51, 40, 5, 12, 20, 58, 57, 16, 1, 57, 58, 
                       34, 17, 52, 25, 44, 20, 41, 50, 58, 53, 47, 58, 14, 40, 39, 
                       10, 53, 1, 34, 54, 53, 49, 60, 31, 54, 35, 39, 23, 10, 55, 
                       12, 56, 57, 48, 57, 15, 35, 23, 13, 10, 22, 4, 25, 17, 47, 
                       5, 36, 51, 6, 44, 26, 27, 40, 58, 59, 15, 55, 19, 21, 7, 
                       32, 45, 13, 41, 47, 30, 36, 1, 60, 43, 48, 61, 51]                                                                                                 
    @test randindex(R.assignments, ref_assignments)[2] == 1.0
end
