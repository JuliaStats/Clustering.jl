using Test
using Clustering

@testset "silhouettes()" begin

local D = [0 1 2 3
           1 0 1 2
           2 1 0 1
           3 2 1 0]

@assert size(D) == (4, 4)

local a = [1, 1, 2, 2]
local c = [2, 2]

@testset "Input checks" begin
    @test_skip silhouettes(a, [1, 1, 2], D) # should throw because cluster counts are inconsistent
    @test_throws ArgumentError silhouettes(a, [3, 2], D)
    @test_throws ArgumentError silhouettes([1, 1, 3, 2], [2, 2], D)
    @test_throws DimensionMismatch silhouettes([1, 1, 2, 2, 2], [2, 3], D)
end

@test @inferred(silhouettes(a, c, D)) ≈ [1.5/2.5, 0.5/1.5, 0.5/1.5, 1.5/2.5]
@test @inferred(silhouettes(a, c, convert(Matrix{Float32}, D))) isa AbstractVector{Float32}
@test silhouettes(a, D) == silhouettes(a, c, D) # c is optional

a = [1, 2, 1, 2]
c = [2, 2]

@test silhouettes(a, c, D) ≈ [0.0, -0.5, -0.5, 0.0]

a = [1, 1, 1, 2]
c = [3, 1]

@test silhouettes(a, c, D) ≈ [0.5, 0.5, -1/3, 0.0]

@testset "silhouettes() handle zero cluster distances correctly" begin
    a = [fill(1, 5); fill(2, 5)]
    d = fill(0, (10, 10))

    @test silhouettes(a, d) == fill(0.0, 10)

    d = fill(1, (10, 10))
    d[1, 2] = d[2, 1] = 5
    
    @test silhouettes(a, d) == [[-0.5, -0.5]; fill(0.0, 8)]
end

@testset "silhouette() throws an error when degenerated clustering is given" begin
    a = fill(1, 10)
    d = fill(1, (10, 10))
    for i in 1:10; d[i, i] = 0; end
    
    @test_throws ArgumentError silhouettes(a, d)
end

end
