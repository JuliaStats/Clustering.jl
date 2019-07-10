using Test

using Distances
using Clustering

@testset "kmedoids() (k-medoids)" begin

@testset "Argument checks" begin
    Random.seed!(34568)
    @test_throws ArgumentError kmedoids(randn(2, 3), 1)
    @test_throws ArgumentError kmedoids(randn(2, 3), 4)
    costs = inv.(max.(pairwise(Euclidean(), randn(2, 3), dims=2), 0.1))
    @test kmedoids(costs, 2) isa KmedoidsResult
    @test_throws ArgumentError kmedoids(costs, 2, display=:mylog)
    for disp in keys(Clustering.DisplayLevels)
        @test kmedoids(costs, 2, display=disp) isa KmedoidsResult
    end
end

Random.seed!(34568)

d = 3
n = 200
k = 10

X = rand(d, n)
costs = pairwise(SqEuclidean(), X, dims=2)
@assert size(costs) == (n, n)

R = kmedoids(costs, k)
@test isa(R, KmedoidsResult)
@test nclusters(R) == k
@test length(R.medoids) == length(unique(R.medoids))
@test all(a -> 1 <= a <= k, R.assignments)
@test R.assignments[R.medoids] == 1:k # Every medoid should belong to its own cluster
@test sum(counts(R)) == n
@test wcounts(R) == counts(R)
@test R.acosts == costs[LinearIndices((n, n))[CartesianIndex.(R.medoids[R.assignments], 1:n)]]
@test isapprox(sum(R.acosts), R.totalcost)
@test R.converged


# this data set has three obvious groups:
# group 1: [1, 3, 4], values: [1, 2, 3]
# group 2: [2, 5, 7], values: [6, 7, 8]
# group 3: [6, 8, 9], values: [21, 20, 22]
#

X = reshape(map(Float64, [1, 6, 2, 3, 7, 21, 8, 20, 22]), 1, 9)
costs = pairwise(SqEuclidean(), X, dims=2)

R = kmedoids!(costs, [1, 2, 6])
@test isa(R, KmedoidsResult)
@test nclusters(R) == 3
@test R.medoids == [3, 5, 6]
@test R.assignments == [1, 2, 1, 1, 2, 3, 2, 3, 3]
@test counts(R) == [3, 3, 3]
@test wcounts(R) == counts(R)
@test R.acosts ≈ [1, 1, 0, 1, 0, 0, 1, 1, 1]
@test R.totalcost ≈ 6.0
@test R.converged

end
