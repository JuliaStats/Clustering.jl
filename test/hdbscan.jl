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
    #test for main function
    @test_throws DomainError hdbscan(data, 5, 0)
    @test_nowarn @inferred(hdbscan(data, 5, 3))

    # tests for result
    result = @inferred(hdbscan(data, 5, 3))
    @test isa(result, HdbscanResult)
    @test sum(length, result.clusters) == size(data, 2)
end