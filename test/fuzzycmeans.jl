using Test
using Clustering

srand(34568)

m = 3
n = 1000
k = 5

x = rand(m,n)

fuzziness = 2.0
srand(34568)
r = fuzzy_cmeans(x, k, fuzziness)
@test isa(r, FuzzyCMeansResult{Float64})
@test size(r.centers) == (m,k)
@test size(r.weights) == (n,k)
@test all([s ≈ 1.0 for s in sum(r.weights, dims=2)])
@test all(0 .<= r.weights .<= 1)

fuzziness = 3.0
srand(34568)
r = fuzzy_cmeans(x, k, fuzziness)
@test isa(r, FuzzyCMeansResult{Float64})
@test size(r.centers) == (m,k)
@test size(r.weights) == (n,k)
@test all([s ≈ 1.0 for s in sum(r.weights, dims=2)])
@test all(0 .<= r.weights .<= 1)
