using Test
using Clustering
using Distances
using Random
using LinearAlgebra

import Distances.pairwise!

# custom distance metric
struct MySqEuclidean <: SemiMetric end

# redefinition of Distances.pairwise! for MySqEuclidean type
function pairwise!(r::AbstractMatrix, dist::MySqEuclidean,
                   a::AbstractMatrix, b::AbstractMatrix; dims::Integer=2)
    dims == 2 || throw(ArgumentError("only dims=2 supported for MySqEuclidean distance"))
    mul!(r, transpose(a), b)
    sa2 = sum(abs2, a, dims=1)
    sb2 = sum(abs2, b, dims=1)
    @inbounds r .= sa2' .+ sb2 .- 2r
end

@testset "kmeans() (k-means)" begin

@testset "Argument checks" begin
    Random.seed!(34568)
    @test_throws ArgumentError kmeans(randn(2, 3), 1)
    @test_throws ArgumentError kmeans(randn(2, 3), 4)
    @test kmeans(randn(2, 3), 2) isa KmeansResult
    @test_throws ArgumentError kmeans(randn(2, 3), 2, display=:mylog)
    for disp in keys(Clustering.DisplayLevels)
        @test kmeans(randn(2, 3), 2, display=disp) isa KmeansResult
    end
end

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
    @test isa(r, KmeansResult{Matrix{Float64}, Float64, Int})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == r.counts
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
    @test isa(r, KmeansResult{Matrix{Float32}, Float32, Int})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == r.counts
    @test sum(r.costs) ≈ r.totalcost

    Random.seed!(34568)
    r_t = kmeans(x32t', k; maxiter=50)
    @test equal_kmresults(r, r_t)
end

@testset "weighted" begin
    w = rand(n)
    Random.seed!(34568)
    r = kmeans(x, k; maxiter=50, weights=w)
    @test isa(r, KmeansResult{Matrix{Float64}, Float64, Float64})
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
    @test isa(r, KmeansResult{Matrix{Float64}, Float64, Int})
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(r.counts) == k
    @test sum(r.counts) == n
    @test r.cweights == r.counts
    @test sum(r.costs) ≈ r.totalcost
    @test equal_kmresults(r, r2)

    Random.seed!(34568)
    r_t = kmeans(xt', k; maxiter=50, init=:kmcen, distance=MySqEuclidean())
    @test equal_kmresults(r, r_t)
end

@testset "Argument checks" begin
    Random.seed!(34568)
    n = 50
    k = 10
    x = randn(m, n)

    @testset "init=" begin
        @test_throws ArgumentError kmeans(x, k, init=1:(k-2))
        @test_throws ArgumentError kmeans(x, k, init=1:(k+2))
        @test kmeans(x, k, init=1:k, maxiter=5) isa KmeansResult

        @test_throws ArgumentError kmeans(x, k, init=:myseeding)
        for algname in (:kmpp, :kmcen, :rand)
            alg = Clustering.seeding_algorithm(algname)
            @test kmeans(x, k, init=algname) isa KmeansResult
            @test kmeans(x, k, init=alg) isa KmeansResult
        end
    end
end

@testset "Integer data" begin
    x = rand(Int16, m, n)
    Random.seed!(654)
    r = kmeans(x, k; maxiter=50)

    @test isa(r, KmeansResult{Matrix{Float64}, Float64, Int})
end

@testset "kmeans! data types" begin
    Random.seed!(1101)
    for TX in (Int, Float32, Float64)
        for TC in (Float32, Float64)
            for TW in (Nothing, Int, Float32, Float64)
                x = rand(TX, m, n)
                c = rand(TC, m, k)
                if TW == Nothing
                    r = kmeans!(x, c; maxiter=1)
                    @test isa(r, KmeansResult{Matrix{TC},<:Real,Int})
                else
                    w = rand(TW, n)
                    r = kmeans!(x, c; weights=w, maxiter=1)
                    @test isa(r, KmeansResult{Matrix{TC},<:Real,TW})
                end
            end
        end
    end
end

end
