using Test
using Clustering

@testset "fuzzy_cmeans()" begin

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
