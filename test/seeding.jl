using Clustering
using Distances: SqEuclidean, pairwise
using Test

@testset "seeding" begin

Random.seed!(34568)

@test RandSeedAlg <: SeedingAlgorithm
@test KmppAlg <: SeedingAlgorithm
@test KmCentralityAlg <: SeedingAlgorithm

alldistinct(x::Vector{Int}) = (length(Set(x)) == length(x))

function min_interdist(X::AbstractMatrix)
    dists = pairwise(SqEuclidean(), X, dims=2)
    n = size(X, 2)
    r = Inf
    for i = 1:n, j = 1:n
        if i != j && dists[i,j] < r
            r = dists[i,j]
        end
    end
    return r
end


d = 3
n = 100
k = 5
X = rand(d, n)
C = pairwise(SqEuclidean(), X, dims=2)

Xt = copy(transpose(X))
Ct = copy(transpose(C))

md0 = min_interdist(X)

@testset "Argument checks" begin
    @test_throws ArgumentError initseeds([1, 2], X, 3)
    @test initseeds([1, 2], X, 2) == [1, 2]
    @test_throws ArgumentError initseeds([-1, 2, 3], X, 3)
    @test_throws ArgumentError initseeds([1, n+2, 3], X, 3)

    @test_throws ArgumentError initseeds_by_costs([1, 2], C, 3)
    @test initseeds_by_costs([1, 2], C, 2) == [1, 2]

    @test_throws ArgumentError initseeds(:myseeding, X, 2)
    iseeds = initseeds(:kmpp, X, k)
    @test_throws DimensionMismatch copyseeds!(Matrix{Float64}(undef, 3, 6), X, iseeds)
    @test_throws DimensionMismatch copyseeds!(Matrix{Float64}(undef, 4, 5), X, iseeds)
    @test copyseeds!(Matrix{Float64}(undef, 3, 5), X, iseeds) isa Matrix{Float64}

    @testset "Seeds number check for $(typeof(alg))" for alg in
            (RandSeedAlg(), KmppAlg(), KmCentralityAlg())
        @test_throws ArgumentError initseeds(alg, X, 0)
        @test_throws ArgumentError initseeds(alg, X, n + 1)
        @test_throws ArgumentError initseeds_by_costs(alg, C, 0)
        @test_throws ArgumentError initseeds_by_costs(alg, C, n + 1)
        @test initseeds(alg, X, 4) isa Vector{Int}
        @test initseeds_by_costs(alg, C, 4) isa Vector{Int}
    end
end

@testset "RandSeed" begin
    Random.seed!(34568)
    iseeds = initseeds(RandSeedAlg(), X, k)
    @test length(iseeds) == k
    @test alldistinct(iseeds)
    Random.seed!(34568)
    iseeds_t = initseeds(RandSeedAlg(), Xt', k)
    @test iseeds == iseeds_t

    Random.seed!(34568)
    iseeds2 = initseeds(:rand, X, k)
    @test iseeds2 == iseeds

    Random.seed!(34568)
    iseeds = initseeds_by_costs(RandSeedAlg(), C, k)
    @test length(iseeds) == k
    @test alldistinct(iseeds)
    Random.seed!(34568)
    iseeds_t = initseeds_by_costs(RandSeedAlg(), Ct', k)
    @test iseeds == iseeds_t

    R = copyseeds!(Matrix{Float64}(undef, d, k), X, iseeds)
    @test isa(R, Matrix{Float64})
    @test R == X[:, iseeds]
    R_t = copyseeds!(Matrix{Float64}(undef, d, k), Xt', iseeds)
    @test R == R_t
end

@testset "Kmpp" begin
    Random.seed!(34568)
    iseeds = initseeds(KmppAlg(), X, k)
    @test length(iseeds) == k
    @test alldistinct(iseeds)
    Random.seed!(34568)
    iseeds_t = initseeds(KmppAlg(), Xt', k)
    @test iseeds == iseeds_t

    Random.seed!(34568)
    iseeds2 = initseeds(:kmpp, X, k)
    @test iseeds2 == iseeds
    Random.seed!(34568)
    iseeds_t2 = initseeds(:kmpp, Xt', k)
    @test iseeds_t2 == iseeds_t

    Random.seed!(34568)
    iseeds = initseeds_by_costs(KmppAlg(), C, k)
    @test length(iseeds) == k
    @test alldistinct(iseeds)
    Random.seed!(34568)
    iseeds_t = initseeds_by_costs(KmppAlg(), Ct', k)
    @test iseeds == iseeds_t

    @test min_interdist(X[:, iseeds]) > 20 * md0
    @test min_interdist((Xt')[:, iseeds]) > 20 * md0

    Random.seed!(34568)
    iseeds = initseeds_by_costs(:kmpp, C, k)
    @test length(iseeds) == k
    @test alldistinct(iseeds)
    Random.seed!(34568)
    iseeds_t = initseeds_by_costs(:kmpp, Ct', k)
    @test iseeds_t == iseeds
end

@testset "Kmcentrality" begin
    Random.seed!(34568)
    iseeds = initseeds(KmCentralityAlg(), X, k)
    @test length(iseeds) == k
    @test alldistinct(iseeds)
    Random.seed!(34568)
    iseeds_t = initseeds(KmCentralityAlg(), Xt', k)
    @test iseeds == iseeds_t

    Random.seed!(34568)
    iseeds2 = initseeds(:kmcen, X, k)
    @test iseeds2 == iseeds
    Random.seed!(34568)
    iseeds_t2 = initseeds(:kmcen, Xt', k)
    @test iseeds_t2 == iseeds_t

    Random.seed!(34568)
    iseeds = initseeds_by_costs(KmCentralityAlg(), C, k)
    @test length(iseeds) == k
    @test alldistinct(iseeds)
    Random.seed!(34568)
    iseeds_t = initseeds_by_costs(KmCentralityAlg(), Ct', k)
    @test iseeds == iseeds_t

    @test min_interdist(X[:, iseeds]) > 2 * md0
    @test min_interdist((Xt')[:, iseeds]) > 2 * md0
end

end
