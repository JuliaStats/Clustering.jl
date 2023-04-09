using Test
using Clustering
using Distances

@testset "get cluster assigments not implemented method" begin

    X = rand(10,3)
    dist = pairwise(SqEuclidean(), X, dims=2)
    R = kmedoids!(dist, [1, 2, 6])

    @test_throws MethodError  assign_clusters(X, R);
end
