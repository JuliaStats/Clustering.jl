using Test
using Clustering
using Distances

@testset "silhouettes()" begin

local D = [0 1 2 3
           1 0 1 2
           2 1 0 1
           3 2 1 0]

@assert size(D) == (4, 4)

@testset "Input checks" begin
    @test_throws DimensionMismatch silhouettes([1, 1, 3, 2], D[1:4, 1:3])
    @test_throws DimensionMismatch silhouettes([1, 1, 2, 2, 2], D)
    @test_throws Exception silhouettes([1, 1, 2, 2, 2], D, batch_size=3)
    D2 = copy(D)
    D2[2, 3] = 4
    @test_throws ArgumentError silhouettes([1, 1, 2, 2], D2)
end

@test @inferred(silhouettes([1, 1, 2, 2], D)) ≈ [1.5/2.5, 0.5/1.5, 0.5/1.5, 1.5/2.5]
@test @inferred(silhouettes([1, 1, 2, 2], convert(Matrix{Float32}, D))) isa AbstractVector{Float32}

@test silhouettes([1, 2, 1, 2], D) ≈ [0.0, -0.5, -0.5, 0.0]
@test silhouettes([1, 1, 1, 2], D) ≈ [0.5, 0.5, -1/3, 0.0]

@testset "zero cluster distances correctly" begin
    a = [fill(1, 5); fill(2, 5)]
    d = fill(0, (10, 10))

    @test silhouettes(a, d) == fill(0.0, 10)

    d = fill(1, (10, 10))
    for i in 1:10; d[i, i] = 0; end
    d[1, 2] = d[2, 1] = 5

    @test silhouettes(a, d) == [[-0.5, -0.5]; fill(0.0, 8)]
end

@testset "throws an error when degenerated clustering is given" begin
    a = fill(1, 10)
    d = fill(1, (10, 10))
    for i in 1:10; d[i, i] = 0; end

    @test_throws ArgumentError silhouettes(a, d)
end

@testset "empty clusters handled correctly (#241)" begin
    X = rand(MersenneTwister(123), 3, 10)
    pd = pairwise(Euclidean(), X, dims=2)
    asgns = [5, 2, 2, 3, 2, 2, 3, 2, 3, 5]
    @test all(>=(-0.5), silhouettes(asgns, pd))
    @test all(>=(-0.5), silhouettes(asgns, X, metric=Euclidean()))
end

@testset "silhouettes(metric=$metric, batch_size=$(batch_size !== nothing ? batch_size : "nothing"))" for
        (metric, batch_size, supported) in [
            (Euclidean(), nothing, true),
            (Euclidean(), 1000, true),
            (Euclidean(), 10, false),

            (SqEuclidean(), nothing, true),
            (SqEuclidean(), 1000, true),
            (SqEuclidean(), 10, true),
        ]

    Random.seed!(123)
    X = rand(3, 100)
    pd = pairwise(metric, X, dims=2)
    a = rand(1:10, size(X, 2))
    kmeans_clu = kmeans(X, 5)
    if supported
        @test silhouettes(a, X; metric=metric, batch_size=batch_size) ≈ silhouettes(a, pd)
        @test silhouettes(kmeans_clu, X; metric=metric, batch_size=batch_size) ≈ silhouettes(kmeans_clu, pd)
    else
        @test_throws Exception silhouettes(a, X; metric=metric, batch_size=batch_size)
        @test_throws Exception silhouettes(kmeans_clu, X; metric=metric, batch_size=batch_size)
    end
end

end
