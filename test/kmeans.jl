using Test
using Clustering
using Distances
using Random
using LinearAlgebra

import Distances.pairwise!

# custom distance metric
mutable struct MySqEuclidean <: SemiMetric end

# redefinition of Distances.pairwise! for MySqEuclidean type
function pairwise!(r::AbstractMatrix, dist::MySqEuclidean,
                   a::AbstractMatrix, b::AbstractMatrix)
    mul!(r, transpose(a), b)
    sa2 = sum(abs2, a, dims=1)
    sb2 = sum(abs2, b, dims=1)
    @inbounds r .= sa2' .+ sb2 .- 2r
end

@testset "kmeans() (k-means)" begin

Random.seed!(34568)

m = 3
n = 1000
k = 10

x = rand(m, n)
xt = copy(transpose(x))

equal_kmresults(km1::KmeansResult, km2::KmeansResult) =
    all(getfield(km1, η) == getfield(km2, η) for η ∈ fieldnames(KmeansResult))

@testset "non-weighted" begin
    Random.seed!(34568)
    r = kmeans(x, k; maxiter=50)
    @test isa(r, KmeansResult{Float64})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == map(Float64, r.counts)
    @test sum(r.costs) ≈ r.totalcost

    Random.seed!(34568)
    r_t = kmeans(xt', k; maxiter=50)
    @test equal_kmresults(r, r_t)
end

@testset "non-weighted (float32)" begin
    Random.seed!(34568)
    x32 = map(Float32, x)
    x32t = copy(x32')
    r = kmeans(x32, k; maxiter=50)
    @test isa(r, KmeansResult{Float32})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == map(Float64, r.counts)
    @test sum(r.costs) ≈ r.totalcost

    Random.seed!(34568)
    r_t = kmeans(x32t', k; maxiter=50)
    @test equal_kmresults(r, r_t)
end

@testset "weighted" begin
    w = rand(n)
    Random.seed!(34568)
    r = kmeans(x, k; maxiter=50, weights=w)
    @test isa(r, KmeansResult{Float64})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n

    cw = zeros(k)
    for i = 1:n
        cw[r.assignments[i]] += w[i]
    end
    @test r.cweights ≈ cw
    @test dot(r.costs, w) ≈ r.totalcost

    Random.seed!(34568)
    r_t = kmeans(xt', k; maxiter=50, weights=w)
    @test equal_kmresults(r, r_t)
end

@testset "custom distance" begin
    Random.seed!(34568)
    r = kmeans(x, k; maxiter=50, init=:kmcen, distance=MySqEuclidean())
    r2 = kmeans(x, k; maxiter=50, init=:kmcen)
    @test isa(r, KmeansResult{Float64})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == map(Float64, r.counts)
    @test sum(r.costs) ≈ r.totalcost
    @test equal_kmresults(r, r2)

    Random.seed!(34568)
    r_t = kmeans(xt', k; maxiter=50, init=:kmcen, distance=MySqEuclidean())
    @test equal_kmresults(r, r_t)
end

x_int = rand(Int16, m, n)

@testset "Integer data" begin
    Random.seed!(654)
    r = kmeans(x_int, k; maxiter=50)
    Random.seed!(654)
    r2 = kmeans(convert(Matrix{Float64}, x_int), k; maxiter=50)

    @test equal_kmresults(r, r2)
end

end
