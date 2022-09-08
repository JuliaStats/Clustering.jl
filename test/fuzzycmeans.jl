using Test
using Clustering
using Random, StableRNGs

@testset "fuzzy_cmeans()" begin

rng = StableRNG(42)

@testset "Argument checks" begin
    @test_throws ArgumentError fuzzy_cmeans(randn(2, 3), 1, 2.0)
    @test_throws ArgumentError fuzzy_cmeans(randn(2, 3), 4, 2.0)
    @test_throws ArgumentError fuzzy_cmeans(randn(2, 3), 2, 1.0)
    for disp in keys(Clustering.DisplayLevels)
        @test fuzzy_cmeans(randn(2, 3), 2, 2.0, tol=0.1, display=disp) isa FuzzyCMeansResult
    end
end

Random.seed!(rng, 34568)

d = 3
n = 1000
k = 5

x = rand(rng, d, n)

@testset "fuzziness = 2.0" begin
    fuzziness = 2.0
    Random.seed!(rng, 34568)
    r = fuzzy_cmeans(x, k, fuzziness; rng=rng)
    @test isa(r, FuzzyCMeansResult{Float64})
    @test nclusters(r) == k
    @test size(r.centers) == (d, k)
    @test size(r.weights) == (n, k)
    @test all(0 .<= r.weights .<= 1)
    @test sum(r.weights, dims=2) ≈ fill(1.0, n)

    @test wcounts(r) isa Vector{Float64}
    @test length(wcounts(r)) == n
    @test all(0 .<= wcounts(r) .<= n)
    @test sum(wcounts(r)) ≈ n
end

@testset "fuzziness = 3.0" begin
    fuzziness = 3.0
    Random.seed!(rng, 34568)
    r = fuzzy_cmeans(x, k, fuzziness, rng=rng)
    @test isa(r, FuzzyCMeansResult{Float64})
    @test nclusters(r) == k
    @test size(r.centers) == (d, k)
    @test size(r.weights) == (n, k)
    @test sum(r.weights, dims=2) ≈ fill(1.0, n)
    @test all(0 .<= r.weights .<= 1)

    @test wcounts(r) isa Vector{Float64}
    @test length(wcounts(r)) == n
    @test all(0 .<= wcounts(r) .<= n)
    @test sum(wcounts(r)) ≈ n
end

@testset "Abstract data matrix" begin
    fuzziness = 2.0
    Random.seed!(rng, 34568)
    r = fuzzy_cmeans(view(x, :, :), k, fuzziness, rng=rng)
    @test isa(r, FuzzyCMeansResult{Float64})
    @test nclusters(r) == k
    @test size(r.centers) == (d, k)
    @test size(r.weights) == (n, k)
    @test sum(r.weights, dims=2) ≈ fill(1.0, n)
    @test all(0 .<= r.weights .<= 1)

    @test wcounts(r) isa Vector{Float64}
    @test length(wcounts(r)) == n
    @test all(0 .<= wcounts(r) .<= n)
    @test sum(wcounts(r)) ≈ n
end

@testset "Float32" begin
    fuzziness = 2.0
    xf32 = convert(Matrix{Float32},x)
    Random.seed!(rng, 34568)
    r = fuzzy_cmeans(xf32, k, fuzziness, rng=rng)
    @test isa(r, FuzzyCMeansResult{Float32})
    @test eltype(r.centers) == Float32
    @test wcounts(r) isa Vector{Float64}
end


end
