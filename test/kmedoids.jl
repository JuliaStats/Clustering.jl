using Test
using Distances
using Clustering

@testset "kmedoids() (k-medoids)" begin

@testset "Argument checks" begin
    Random.seed!(34568)
    @test_throws ArgumentError kmedoids(randn(2, 3), 1)
    @test_throws ArgumentError kmedoids(randn(2, 3), 4)
    dist = max.(pairwise(Euclidean(), randn(2, 3), dims=2), 0.1)
    @test @inferred(kmedoids(dist, 2)) isa KmedoidsResult
    # incorrect distance matrix
    invdist = inv.(max.(pairwise(Euclidean(), randn(2, 3), dims=2), 0.1))
    @test_throws ArgumentError kmedoids(invdist, 2)

    @test_throws ArgumentError kmedoids(dist, 2, display=:mylog)
    for disp in keys(Clustering.DisplayLevels)
        @test @inferred(kmedoids(dist, 2, display=disp)) isa KmedoidsResult
    end
end

Random.seed!(34568)

d = 3
n = 200
k = 10

X = rand(d, n)
dist = pairwise(SqEuclidean(), X, dims=2)
@assert size(dist) == (n, n)

Random.seed!(34568)  # reset seed again to known state
R = @inferred(kmedoids(dist, k))
@test isa(R, KmedoidsResult)
@test nclusters(R) == k
@test length(R.medoids) == length(unique(R.medoids))
@test all(a -> 1 <= a <= k, R.assignments)
@test R.assignments[R.medoids] == 1:k # Every medoid should belong to its own cluster
@test sum(counts(R)) == n
@test wcounts(R) == counts(R)
@test R.costs == dist[LinearIndices((n, n))[CartesianIndex.(R.medoids[R.assignments], 1:n)]]
@test isapprox(sum(R.costs), R.totalcost)
@test R.converged

@testset "Support for arrays other than Matrix{T}" begin
    @testset "$(typeof(M))" for M in equivalent_matrices(dist)
        Random.seed!(34568)  # restore seed as kmedoids is not determantistic
        R2 = kmedoids(M, k)
        @test R2.assignments == R.assignments
    end
end

@testset "Duplicated points (#231)" begin
    pts = [0.0 0.0]
    dists = pairwise(SqEuclidean(), pts, dims=2)
    dupmed = kmedoids(dists, 2)
    @test nclusters(dupmed) == 2
    @test sort(dupmed.medoids) == [1, 2]
    @test sort(dupmed.assignments) == [1, 2]
end

@testset "Toy example #1" begin
    pts = [1 2 3; .1 .2 .3; 4 5.6 7]
    # k=1 and k=n cases
    dists = pairwise(SqEuclidean(), pts, dims=2)

    @testset "k=1" begin
        kmed1 = @inferred(kmedoids(dists, 1))
        @test nclusters(kmed1) == 1
        @test assignments(kmed1) == [1, 1, 1]
        @test kmed1.medoids == [2]
    end

    @testset "k=3" begin
        kmed3 = @inferred(kmedoids(dists, 3))
        @test nclusters(kmed3) == 3
        @test sort(assignments(kmed3)) == [1, 2, 3]
        @test sort(kmed3.medoids) == [1, 2, 3]
    end
end

@testset "Toy example #2" begin
    pts = reshape(map(Float64, [1, 6, 2, 3, 7, 21, 8, 20, 22]), 1, 9)
    # this data set has three obvious groups:
    # group 1: [1, 3, 4], values: [1, 2, 3]
    # group 2: [2, 5, 7], values: [6, 7, 8]
    # group 3: [6, 8, 9], values: [21, 20, 22]

    dists = pairwise(SqEuclidean(), pts, dims=2)

    R = @inferred(kmedoids!(dists, [1, 2, 6]))
    @test isa(R, KmedoidsResult)
    @test nclusters(R) == 3
    @test R.medoids == [3, 5, 6]
    @test R.assignments == [1, 2, 1, 1, 2, 3, 2, 3, 3]
    @test counts(R) == [3, 3, 3]
    @test wcounts(R) == counts(R)
    @test R.costs ≈ [1, 1, 0, 1, 0, 0, 1, 1, 1]
    @test R.totalcost ≈ 6.0
    @test R.converged
end

end
