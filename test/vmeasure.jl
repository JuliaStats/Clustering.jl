using Clustering
using Test
import Random

@testset "v-measure" begin

# Tests are taken from the fig. 2 of the referenced paper:
# V-Measure: A conditional entropy-based external cluster evaluation measure,
# Andrew Rosenberg and Julia Hirschberg

clus = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
v = vmeasure(clus, clus)
@test v == 1.0

clas = [1, 1, 1, 2, 3, 3, 3, 3, 1, 2, 2, 2, 2, 1, 3]
v = vmeasure(clas, clus)
@test v ≈ 0.14 atol=1e-2

clas = [1, 1, 1, 2, 2, 3, 3, 3, 1, 1, 2, 2, 2, 3, 3]
v = vmeasure(clas, clus)
@test v ≈ 0.39 atol=1e-2

clus = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6]
clas = [1, 1, 1, 2, 2, 3, 3, 3, 1, 1, 2, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3]
v = vmeasure(clas, clus)
@test v ≈ 0.30 atol=1e-2

clus = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9]
v = vmeasure(clas, clus)
@test v ≈ 0.41 atol=1e-2

@test_throws ArgumentError vmeasure(clas, clus, β = -1.0)

Random.seed!(345618)
m = 3
n = 1000
k = 10
x = rand(m, n)

# non-weighted
r1 = kmeans(x, k; maxiter=50)
r2 = kmeans(x, k; maxiter=50)
v = vmeasure(r1, r2)
@test v ≈ 0.755 atol=1e-3

end
