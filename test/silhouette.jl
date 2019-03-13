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

end
