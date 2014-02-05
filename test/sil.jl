using Base.Test

using Distance
using Clustering


x = rand(100, 500)
dist = pairwise(Euclidean(), x)

assignments = mod(rand(Int, 500), 4) .+ 1

@test isa(silhouettes(assignments, dist), Vector)
