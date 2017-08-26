# simple program to test the new k-means (not ready yet)

using Base.Test
using Clustering
using Distances

import Distances.pairwise!

srand(34568)

m = 3
n = 1000
k = 10

x = rand(m, n)

# non-weighted
r = kmeans(x, k; maxiter=50, n_init=2)
@test isa(r, KmeansResult{Float64})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n
@test r.cweights == map(Float64, r.counts)
@test isapprox(sum(r.costs), r.totalcost)

# non-weighted (float32)
r = kmeans(@compat(map(Float32, x)), k; maxiter=50, n_init=2)
@test isa(r, KmeansResult{Float32})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n
@test r.cweights == map(Float64, r.counts)
@test isapprox(sum(r.costs), r.totalcost)

# weighted
w = rand(n)
r = kmeans(x, k; maxiter=50, weights=w, n_init=2)
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
@test isapprox(r.cweights, cw)

@test isapprox(dot(r.costs, w), r.totalcost)


# custom distance metric
type MySqEuclidean <: SemiMetric end

# redefinition of Distances.pairwise! for MySqEuclidean type
function pairwise!(r::AbstractMatrix, dist::MySqEuclidean, a::AbstractMatrix, b::AbstractMatrix)
    At_mul_B!(r, a, b)
    sa2 = sum(abs2, a, 1)
    sb2 = sum(abs2, b, 1)
    for j = 1 : size(r,2)
        sb = sb2[j]
        @simd for i = 1 : size(r,1)
            @inbounds r[i,j] = sa2[i] + sb - 2 * r[i,j]
        end
    end
    r
end

r = kmeans(x, k; maxiter=50, init=:kmcen, distance=MySqEuclidean())
r2 = kmeans(x, k; maxiter=50, init=:kmcen)
@test isa(r, KmeansResult{Float64})
@test size(r.centers) == (m, k)
@test length(r.assignments) == n
@test all(r.assignments .>= 1) && all(r.assignments .<= k)
@test length(r.costs) == n
@test length(r.counts) == k
@test sum(r.counts) == n
@test r.cweights == map(Float64, r.counts)
@test isapprox(sum(r.costs), r.totalcost)
for fn in fieldnames(r)
    @test getfield(r, fn) == getfield(r2, fn)
end
