using Test
using Clustering

@testset "HDBSCAN" begin
    # make moons for test
    upper_x = [i for i in 0:π/50:π]
    lower_x = [i for i in π/2:π/50:3/2*π]
    upper_y = sin.(upper_x) .+ rand(50)./10
    lower_y = cos.(lower_x) .+ rand(51)./10
    data = hcat([lower_x; upper_x], [lower_y; upper_y])
    #test for main function
    @test_throws DomainError hdbscan(data, 5, 0)
    @test_throws ArgumentError hdbscan(data, 5, 3; gen_mst=false)
    @test_nowarn result = hdbscan(data, 5, 3)

    # tests for result
    result = hdbscan(data, 5, 3)
    @test sum([length(c) for c in result.clusters]) == 101
end