using Test
using Clustering
using Distances

@testset "get cluster assigments not implemented method" begin

    X = rand(10,5)
    dist = pairwise(SqEuclidean(), X, dims=2)
    R = kmedoids!(dist, [1, 2, 3])

    @test_throws MethodError  assign_clusters(X, R);
end
