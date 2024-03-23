using Test
include("../src/hdbscan.jl")

@testset "HDBSCAN" begin
    # test for util functions
    # erf function tests
    @test erf(0) == 0
    @test erf(-0) == -0
    @test isapprox(erf(1.0), 0.8427007929497148)
    @test isapprox(erf(-1.0), -0.8427007929497148)
    @test isapprox(erf(2.4), 0.999311486103355)
    @test isapprox(erf(-2.4), -0.999311486103355)

    # test for HDBSCANGraph
    graph = HDBSCANGraph(4)
    add_edge(graph, (1, 3, 0.1))
    @test graph[1][1] == (3, 0.1) && graph[3][1] == (1, 0.1)

    # test for UnionFind
    uf = UnionFind(10)
    @test group(uf, 1) == 1
    @test sort(members(uf, 1)) == [1]
    @test root(uf, 3) == 3
    @test issame(uf, 1, 2) == false
    @test size(uf, 1) == 1
    @test unite!(uf, 1, 2) == true
    @test issame(uf, 1, 2) == true
    @test size(uf, 1) == 2
    @test size(uf, 2) == 2
    @test unite!(uf, 1, 2) == false
    
    # make moons for test
    upper_x = [i for i in 0:π/50:π]
    lower_x = [i for i in π/2:π/50:3/2*π]
    upper_y = sin.(upper_x) .+ rand(50)./10
    lower_y = cos.(lower_x) .+ rand(51)./10
    data = hcat([lower_x; upper_x], [lower_y; upper_y])
    #test for main function
    @test_throws DomainError hdbscan!(data, 5, 0)
    @test_throws ArgumentError hdbscan!(data, 5, 3; gen_mst=false)
    @test_nowarn result = hdbscan!(data, 5, 3)
end