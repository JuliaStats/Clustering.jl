using Base.Test

using Distance
using Clustering


x = rand(100, 500)
dist = pairwise(Euclidean(), x)

result = kmedoids(dist, 30)
@test isa(result, Clustering.KmedoidsResult)
# There should be no duplicate medoids
@test length(result.medoids) == length(unique(result.medoids))
# Every medoid should belong to its own cluster
for i = 1:30
    @test result.assignments[result.medoids[i]] == i
end

@test_throws ErrorException kmedoids(dist, 500)
