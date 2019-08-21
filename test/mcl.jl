# simple program to test MCL clustering

using Test
using Clustering

@testset "MCL" begin

@testset "Argument Checks" begin
    Random.seed!(34568)
    @test_throws DimensionMismatch mcl(zeros(Float64, 4, 3)) # nonsquare
    adj = inv.(max.(pairwise(Euclidean(), randn(2, 3), dims=2), 0.1))
    @test_throws ArgumentError mcl(zeros(Float64, 3, 3), display=:mylog)
    for disp in keys(Clustering.DisplayLevels)
        @test mcl(zeros(Float64, 3, 3), display=disp) isa MCLResult
    end
end

Random.seed!(34568)

# initialize adjacency matrix of a weighted graph
nodes = [:bat, :bit, :cat, :fit, :hat, :hit]
edges = Tuple{Symbol, Symbol, Float64}[(:cat, :hat, 0.2), (:hat, :bat, 0.16),
                                      (:bat, :cat, 1.0), (:bat, :bit, 0.125),
                                      (:bit, :fit, 0.25), (:fit, :hit, 0.5),
                                      (:hit, :bit, 0.16)]
adj_matrix = zeros(Float64, length(nodes), length(nodes))
for edge in edges
    n1 = findfirst(isequal(edge[1]), nodes)
    n2 = findfirst(isequal(edge[2]), nodes)
    adj_matrix[n1, n2] = adj_matrix[n2, n1] = edge[3]
end
@assert issymmetric(adj_matrix)

@testset "fractional inflation param (1.8)" begin
    res = mcl(adj_matrix, display=:none, inflation=1.8)
    @test isa(res, MCLResult)
    local k = length(res.counts)
    # @show k
    @test k == 2
    @test all(a -> 1 <= a <= k, res.assignments)
    @test length(res.assignments) == length(nodes)
    @test length(res.counts) == k
    local c
    for c in 1:k
        @test count(==(c), res.assignments) == res.counts[c]
    end
    @test res.nunassigned == 0
    @test res.assignments == [1, 2, 1, 2, 1, 2]
end

@testset "integer inflation param (2)" begin
    res = mcl(adj_matrix, display=:none, inflation=2)
    @test isa(res, MCLResult)
    @test length(res.assignments) == length(nodes)
    @test res.nunassigned == 0
end

@testset "test non-integral expansion" begin
    # should not raise an exception
    res = mcl(adj_matrix, display=:none, inflation=1.5, expansion=1.5, save_final_matrix=true)
    @test isa(res, MCLResult)
    @test length(res.assignments) == length(nodes)
    @test size(res.mcl_adj) == size(adj_matrix) # test that the matrix is returned
end

@testset "allow_singles option" begin
    res = mcl(diagm(0 => [1.0, 1.0]), display=:none, allow_singles=true)
    @test length(res.counts) == 2
    @test res.assignments == [1, 2]
    @test res.counts == [1, 1]
    @test res.nunassigned == 0

    res = mcl(diagm(0 => [1.0, 1.0]), display=:none, allow_singles=false)
    @test length(res.counts) == 0
    @test res.assignments == [0, 0]
    @test res.nunassigned == 2
end

@testset "sparse input matrix" begin
    res = mcl(sparse(adj_matrix), display=:none, expansion=2)
    @test isa(res, MCLResult)
    @test length(res.assignments) == length(nodes)
    @test res.nunassigned == 0
    @test eltype(res.mcl_adj) === Float64

    # fractional powers not supported for sparse matrices
    @test_broken mcl(sparse(adj_matrix), display=:none, expansion=2.1)
end

@testset "use Float32 input" begin
    res = mcl(convert(Matrix{Float32},adj_matrix), display=:none, expansion=2)
    @test isa(res, MCLResult)
    @test eltype(res.mcl_adj) === Float32
end

end
