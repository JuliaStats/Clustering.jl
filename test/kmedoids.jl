using Base.Test

using Distance
using Clustering


x = rand(100, 500)
dist = pairwise(Euclidean(), x)

@test isa(kmedoids(dist, 3), Clustering.KmedoidsResult)

@test_throws kmedoids(dist, 500)
