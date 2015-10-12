# simple program to test the new k-means (not ready yet)

using Base.Test
using Clustering

srand(34568)

m = 3
n = 1000
k = 10

x = rand(m, n)

# non-weighted
r = kmeans(x, k; maxiter=50)
@test isa(r, KmeansResult{Float64})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n
@test r.cweights == map(Float64, r.counts)
@test_approx_eq sum(r.costs) r.totalcost

# non-weighted (float32)
r = kmeans(map(Float32, x), k; maxiter=50)
@test isa(r, KmeansResult{Float32})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n
@test r.cweights == map(Float64, r.counts)
@test_approx_eq sum(r.costs) r.totalcost

# weighted
w = rand(n)
r = kmeans(x, k; maxiter=50, weights=w)
@test isa(r, KmeansResult{Float64})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n

cw = zeros(k)
for i = 1:n
	cw[r.assignments[i]] += w[i]
end
@test_approx_eq r.cweights cw

@test_approx_eq dot(r.costs, w) r.totalcost

