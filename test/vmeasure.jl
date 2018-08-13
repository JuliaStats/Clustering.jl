using Test
using Clustering

@testset "V-measure" begin
    @testset "reproducing fig.2" begin
        # Tests are taken from the fig. 2 of the referenced paper:
        # V-Measure: A conditional entropy-based external cluster evaluation measure,
        # Andrew Rosenberg and Julia Hirschberg

        clus = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        v = vmeasure(clus, clus)
        @test v == 1.0

        clas = [1, 1, 1, 2, 3, 3, 3, 3, 1, 2, 2, 2, 2, 1, 3]
        v = vmeasure(clas, clus)
        @test v ≈ 0.14 atol=1e-2

        clas = [1, 1, 1, 2, 2, 3, 3, 3, 1, 1, 2, 2, 2, 3, 3]
        v = vmeasure(clas, clus)
        @test v ≈ 0.39 atol=1e-2

        clus = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6]
        clas = [1, 1, 1, 2, 2, 3, 3, 3, 1, 1, 2, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3]
        v = vmeasure(clas, clus)
        @test v ≈ 0.30 atol=1e-2

        clus = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9]
        v = vmeasure(clas, clus)
        @test v ≈ 0.41 atol=1e-2

        @test_throws ArgumentError vmeasure(clas, clus, β = -1.0)
    end

    @testset "comparing 2 k-means clusterings" begin
        Random.seed!(34568)
        m = 3
        n = 1000
        k = 10
        x = rand(m, n)

        # non-weighted
        r1 = kmeans(x, k; maxiter=50)
        r2 = kmeans(x, k; maxiter=50)
        v = vmeasure(r1, r2)
        @test 0.5 < v < 1.0
        @test_broken v ≈ 0.75 atol=1e-2 # FIXME why 0.75?
    end

    @testset "comparing 2 random label assignments" begin
        Random.seed!(34568)
        k = 10
        n = 10000

        a1 = rand(1:k, n)
        a2 = rand(1:k, n)
        v = vmeasure(a1, a2)
        @test v ≈ 0.0 atol=1e-2 # should be close to zero
    end
end
