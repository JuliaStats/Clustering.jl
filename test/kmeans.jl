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
    @test_throws ArgumentError kmeans(randn(2, 3), 0)
    @test_throws ArgumentError kmeans(randn(2, 3), 4)
    @test kmeans(randn(2, 3), 2) isa KmeansResult
    @test_throws ArgumentError kmeans(randn(2, 3), 2, display=:mylog)
    for disp in keys(Clustering.DisplayLevels)
        @test kmeans(randn(2, 3), 2, display=disp) isa KmeansResult
    end
end

@testset "k=1 and k=n corner cases" begin
    x = [0.5 1 2; 1 0.5 0; 3 2 1]
    km1 = kmeans(x,1)
    @test km1.centers == reshape([7/6, 0.5, 2.0], (3, 1))
    @test km1.counts == [3]
    @test km1.assignments == [1, 1, 1]
    km3 = kmeans(x, 3)
    @test km3.centers == x
    @test km3.counts == fill(1, (3))
    @test km3.assignments == 1:3
    w = [0.5, 2.0, 1.0]
    @test kmeans(x,1,weights=w).wcounts == [3.5]
    @test kmeans(x,3,weights=w).wcounts == w
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
    @test nclusters(r) == k
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(counts(r)) == k
    @test sum(counts(r)) == n
    @test wcounts(r) == counts(r)
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
    @test nclusters(r) == k
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(counts(r)) == k
    @test sum(counts(r)) == n
    @test wcounts(r) == counts(r)
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
    @test nclusters(r) == k
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(counts(r)) == k
    @test sum(counts(r)) == n

    cw = zeros(k)
    for i = 1:n
        cw[r.assignments[i]] += w[i]
    end
    @test wcounts(r) ≈ cw
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
    @test nclusters(r) == k
    @test size(r.centers) == (m, k)
    @test length(r.assignments) == n
    @test all(a -> 1 <= a <= k, r.assignments)
    @test length(r.costs) == n
    @test length(counts(r)) == k
    @test sum(counts(r)) == n
    @test wcounts(r) == r.counts
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
