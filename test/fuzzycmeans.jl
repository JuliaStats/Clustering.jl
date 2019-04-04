using Test
using Clustering

@testset "fuzzy_cmeans()" begin

@testset "Argument checks" begin
    Random.seed!(34568)
    @test_throws ArgumentError fuzzy_cmeans(randn(2, 3), 1, 2.0)
    @test_throws ArgumentError fuzzy_cmeans(randn(2, 3), 4, 2.0)
    @test_throws ArgumentError fuzzy_cmeans(randn(2, 3), 2, 1.0)
    for disp in keys(Clustering.DisplayLevels)
        @test fuzzy_cmeans(randn(2, 3), 2, 2.0, tol=0.1, display=disp) isa FuzzyCMeansResult
    end
end

Random.seed!(34568)

m = 3
n = 1000
k = 5

x = rand(m,n)

@testset "fuzziness = 2.0" begin
    fuzziness = 2.0
    Random.seed!(34568)
    r = fuzzy_cmeans(x, k, fuzziness)
    @test isa(r, FuzzyCMeansResult{Float64})
    @test size(r.centers) == (m,k)
    @test size(r.weights) == (n,k)
    @test sum(r.weights, dims=2) ≈ fill(1.0, n)
    @test all(0 .<= r.weights .<= 1)
end

@testset "fuzziness = 3.0" begin
    fuzziness = 3.0
    Random.seed!(34568)
    r = fuzzy_cmeans(x, k, fuzziness)
    @test isa(r, FuzzyCMeansResult{Float64})
    @test size(r.centers) == (m,k)
    @test size(r.weights) == (n,k)
    @test sum(r.weights, dims=2) ≈ fill(1.0, n)
    @test all(0 .<= r.weights .<= 1)
end

@testset "Abstract data matrix" begin
    fuzziness = 2.0
    Random.seed!(34568)
    r = fuzzy_cmeans(view(x, :, :), k, fuzziness)
    @test isa(r, FuzzyCMeansResult{Float64})
    @test size(r.centers) == (m,k)
    @test size(r.weights) == (n,k)
    @test sum(r.weights, dims=2) ≈ fill(1.0, n)
    @test all(0 .<= r.weights .<= 1)
end

end
