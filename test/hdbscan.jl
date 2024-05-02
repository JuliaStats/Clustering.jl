using Test
using Clustering
using Clustering: UnionFind, set_id, items, root, setsize, unite!

@testset "UnionFind" begin
    uf = UnionFind(10)
    @test set_id(uf, 5) == 5
    @test items(uf, 5) == [5]
    @test !(unite!(uf, 5, 5))
    @test unite!(uf, 4, 5)
    @test setsize(uf, 4) == 2
    @test root(uf, 5) == 4
end

@testset "HDBSCAN" begin
    # make moons for test
    upper_x = [i for i in 0:π/50:π]
    lower_x = [i for i in π/2:π/50:3/2*π]
    upper_y = sin.(upper_x) .+ rand(50)./10
    lower_y = cos.(lower_x) .+ rand(51)./10
    data = hcat([lower_x; upper_x],
                [lower_y; upper_y])'
    # test for main function
    @test_throws DomainError hdbscan(data, 5, 0)
    @test_nowarn @inferred(hdbscan(data, 5, 10))

    # tests for result
    result = @inferred(hdbscan(data, 5, 15))
    @test isa(result, HdbscanResult)
    @test sum(length, result.clusters) == size(data, 2)
    # test whether the model recognizes the moons
    @test length(result.clusters) == 3

    # generate more complicated data
    n_points = 100  # number of points
    clusters = 4    # number of clusters

    # generate the centers of clusters
    centers = [(rand(1:10), rand(1:10)) for _ in 1:clusters]

    # generate all the points
    data = hcat([0.5 * randn(2, n_points) .+ centers[i] for i in 1:clusters]...)
    # check the effect of min_size
    nclusters1 = length(@inferred(hdbscan(data, 5, 5)).clusters)
    nclusters2 = length(@inferred(hdbscan(data, 5, 15)).clusters)
    nclusters3 = length(@inferred(hdbscan(data, 5, 25)).clusters)
    @test nclusters1 >= nclusters2
    @test nclusters2 >= nclusters3
end