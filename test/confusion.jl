# Test confusion matrix

using Test
using Clustering

@testset "confusion() (Confusion matrix)" begin

    @testset "small size tests" begin
        @test confusion([0,0,0], [0,0,0]) == [3 0; 0 0]
        @test confusion([0,0,1], [0,0,0]) == [1 0; 2 0]
        @test confusion([0,1,1], [0,0,0]) == [1 0; 2 0]
        @test confusion([1,1,1], [0,0,0]) == [3 0; 0 0]

        @test confusion([0,0,0], [0,0,1]) == [1 2; 0 0]
        @test confusion([0,0,1], [0,0,1]) == [1 0; 0 2]
        @test confusion([0,1,1], [0,0,1]) == [0 1; 1 1]
        @test confusion([1,1,1], [0,0,1]) == [1 2; 0 0]

        @test confusion([0,0,0], [0,1,1]) == [1 2; 0 0]
        @test confusion([0,0,1], [0,1,1]) == [0 1; 1 1]
        @test confusion([0,1,1], [0,1,1]) == [1 0; 0 2]
        @test confusion([1,1,1], [0,1,1]) == [1 2; 0 0]

        @test confusion([0,0,0], [1,1,1]) == [3 0; 0 0]
        @test confusion([0,0,1], [1,1,1]) == [1 0; 2 0]
        @test confusion([0,1,1], [1,1,1]) == [1 0; 2 0]
        @test confusion([1,1,1], [1,1,1]) == [3 0; 0 0]

    end

    @testset "specifying element type" begin
        @test @inferred(confusion(Int, [1,1,1], [1,1,1])) isa Matrix{Int}
        @test @inferred(confusion(Float64, [1,1,1], [1,1,1])) isa Matrix{Float64}
    end

    @testset "comparing 2 k-means clusterings" begin
        m = 3
        n = 100
        k = 1
        x = rand(m, n)

        # non-weighted
        r1 = kmeans(x, k; maxiter=5)
        r2 = kmeans(x, k; maxiter=5)
        C = confusion(r1, r2)
        @test C == [n*(n-1)/2 0; 0 0]

        C = confusion(Float64, r1, r2)
        @test C == [n*(n-1)/2 0; 0 0]
    end

end

