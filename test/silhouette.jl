using Test
using Clustering
using Distances

@testset "silhouettes()" begin

local D = [0 1 2 3
           1 0 1 2
           2 1 0 1
           3 2 1 0]

@assert size(D) == (4, 4)

local a = [1, 1, 2, 2]
local c = [2, 2]

@testset "Input checks" begin
    @test_skip silhouettes(a, D; counts=[1, 1, 2]) # should throw because cluster counts are inconsistent
    @test_throws ArgumentError silhouettes(a, D; counts=[3, 2])
    @test_throws ArgumentError silhouettes([1, 1, 3, 2], D; counts=[2, 2])
    @test_throws DimensionMismatch silhouettes([1, 1, 2, 2, 2], D; counts=[2, 3])
end

@test @inferred(silhouettes(a, D; counts=c)) ≈ [1.5/2.5, 0.5/1.5, 0.5/1.5, 1.5/2.5]
@test @inferred(silhouettes(a, convert(Matrix{Float32}, D); counts=c)) isa AbstractVector{Float32}
@test silhouettes(a, D) == silhouettes(a, D; counts=c) # c is optional

a = [1, 2, 1, 2]
c = [2, 2]

@test silhouettes(a, D; counts=c) ≈ [0.0, -0.5, -0.5, 0.0]

a = [1, 1, 1, 2]
c = [3, 1]

@test silhouettes(a, D; counts=c) ≈ [0.5, 0.5, -1/3, 0.0]

@testset "zero cluster distances correctly" begin
    a = [fill(1, 5); fill(2, 5)]
    d = fill(0, (10, 10))

    @test silhouettes(a, d) == fill(0.0, 10)

    d = fill(1, (10, 10))
    for i in 1:10; d[i, i] = 0; end
    d[1, 2] = d[2, 1] = 5

    @test silhouettes(a, d) == [[-0.5, -0.5]; fill(0.0, 8)]
end

@testset "throws an error when degenerated clustering is given" begin
    a = fill(1, 10)
    d = fill(1, (10, 10))
    for i in 1:10; d[i, i] = 0; end

    @test_throws ArgumentError silhouettes(a, d)
end

@testset "empty clusters handled correctly (#241)" begin
    X = rand(MersenneTwister(123), 10, 5)
    pd = pairwise(Euclidean(), X, dims=1)
    @test all(>=(-0.5), silhouettes([5, 2, 2, 3, 2, 2, 3, 2, 3, 5], pd))
end

@testset "streaming silhouettes" begin
    import Clustering: sil_aggregate_distances_normalized_streaming, sil_aggregate_distances_normalized, silhouettes_using_cache, silhouettes_cache
    @testset "$metric" for metric in [SqEuclidean(), CosineDist()]
        nclusters = 10
        dims = 3
        n = 100
        X = rand(MersenneTwister(123), dims, n)
        pd = pairwise(metric, X, dims=2)
        a = rand(1:nclusters, n)
        stats1 = @timed s_standard = silhouettes(a, pd)
        stats2 = @timed s_streaming_at_once = silhouettes(a, X; metric=metric, nclusters=nclusters, method=:cached)
        
        @test isapprox(s_standard, s_streaming_at_once)
        
        pre_at_init = silhouettes_cache(eltype(X), metric, nclusters, dims)
        pre = silhouettes_cache(eltype(X), metric, nclusters, dims)
        pre_all_at_once = silhouettes_cache(eltype(X), metric, nclusters, dims)
        batch_size = 10
        for (x, aa) in zip(eachslice(reshape(X, dims, batch_size, trunc(Int, n/batch_size)), dims=3), 
                           eachslice(reshape(a, batch_size, trunc(Int, n/batch_size)), dims=2))
            pre = pre(x, aa)
        end
        pre_all_at_once = pre_all_at_once(X, a)
        # counts sanity test
        @test sum(pre.counts) == n
        # make sure the batched calculation is the same as calculating all at once and different than initialization
        for prop in propertynames(pre)
            @test isapprox(getproperty(pre, prop), getproperty(pre_all_at_once, prop))
            prop in [:nclusters, :dims] && continue
            @test !isapprox(getproperty(pre, prop), getproperty(pre_at_init, prop))
        end
    
        # compare with standard calculation results
        r_standard = sil_aggregate_distances_normalized(a, reshape(pre.counts, :), pd)
        r_streaming = sil_aggregate_distances_normalized_streaming(X, a, pre)
        @test isapprox(r_standard, r_streaming)
    
        s_streaming = []
        for (x, aa) in zip(eachslice(reshape(X, dims, batch_size, trunc(Int, n/batch_size)), dims=3), 
                           eachslice(reshape(a, batch_size, trunc(Int, n/batch_size)), dims=2))
            s_streaming = vcat(s_streaming, silhouettes_using_cache(x, aa, pre))
        end
        # compare final scores with standard calculation results
        @test isapprox(s_standard, s_streaming) 
    end
end

end
