# simple program to test dp-means

using Base.Test
using Clustering

srand(34568)

m = 3
n = 500

# simple easily-separable case
x = [rand(m, n) 3+rand(m,n)]
λ = 2.0

res = dpmeans(x, λ; maxiter=50)
@test isa(res, DPmeansResult{Float64})
@test size(res.centers) == (m, res.k)
@test length(res.assignments) == size(x,2)
@test all(res.assignments .>= 1) && all(res.assignments .<= 2)
@test length(res.costs) == size(x,2)
@test length(res.counts) == res.k
@test sum(res.counts) == size(x,2)
@test_approx_eq sum(res.costs) res.totalcost
@test res.converged

# float32 example
X = convert(Matrix{Float32}, 
	[0.0 1.0  0.0;
     1.0 0.0  0.5
     0.0 1.0 10.0] )
λ = 3

res = dpmeans(X, λ; maxiter=50)
@test isa(res, DPmeansResult{Float32})
@test size(res.centers) == (size(X,1), res.k)
@test length(res.assignments) == size(X,2)
@test all(res.assignments .>= 1) && all(res.assignments .<= 2)
@test length(res.costs) == size(X,2)
@test length(res.counts) == res.k
@test sum(res.counts) == size(X,2)
@test_approx_eq sum(res.costs) res.totalcost
@test res.converged

