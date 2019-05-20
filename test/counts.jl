# Test cross-tabulation

# simple ClusteringResult subtype for testing
struct SimpleCluRes{T} <: ClusteringResult
    assignments::Vector{T}
    counts::Vector{Int}

    SimpleCluRes(assignments::AbstractVector{T}) where {T<:Integer} =
        new{T}(assignments,
               isempty(assignments) ? Vector{Int}() :
               counts(assignments, (1:maximum(assignments))))
end

@testset "counts() (contingency matrix)" begin

# StatsBase's counts() doesn't allow empty inputs
@test_throws ArgumentError counts(Int[], Int[])
# Clustering's counts()
@test counts(SimpleCluRes(Int[]), Int[]) == Matrix{Int}(undef, 0, 0)
@test_throws DimensionMismatch counts(SimpleCluRes([1]), Int[])
@test_throws DimensionMismatch counts(SimpleCluRes([1]), [2, 1])
#@test_throws ArgumentError counts([1, 1], [0, 1]) # doesn't throw as StatsBase.counts() is called
#@test_throws ArgumentError counts([1, -1], [2, 1]) # doesn't throw as StatsBase.counts() is called
@test_throws ArgumentError counts(SimpleCluRes([1, 1]), [0, 1])
@test_throws ArgumentError counts(SimpleCluRes([1, -1]), [2, 1])

# supports different Integer types
@test counts(SimpleCluRes(Int32[1, 2]), Int16[1, 2]) == Matrix{Int}(I, 2, 2)

@test counts([1, 2, 3], [1, 2, 3]) == Matrix{Int}(I, 3, 3)
@test counts([2, 3, 1], [1, 2, 3]) == [0 0 1; 1 0 0; 0 1 0]

# 3rd cluster in A missing
@test counts([2, 4, 1], [1, 2, 3]) == [0 0 1; 1 0 0; 0 0 0; 0 1 0]
@test counts(SimpleCluRes([2, 4, 1]), [1, 2, 3]) == [0 0 1; 1 0 0; 0 0 0; 0 1 0]

# 1st cluster in B missing (StatsBase.counts() and Clustering.counts() give different results)
@test counts([2, 3, 1], [2, 2, 3]) == [0 1; 1 0; 1 0]
@test counts(SimpleCluRes([2, 3, 1]), [2, 2, 3]) == [0 0 1; 0 1 0; 0 1 0]

@test counts([2, 2, 1], [1, 1, 1]) == reshape([1; 2], (2, 1))
@test counts([1, 1, 1], [2, 2, 1]) == [1 2]

@testset "with ClusteringResult objects" begin
    Random.seed!(34568)
    X = rand(3, 25)

    clu3 = kmeans(X, 3)
    clu5 = kmeans(X, 5)

    Y = rand(3, 20)
    cluY = kmeans(Y, 3)

    @test_throws DimensionMismatch counts(clu3, cluY)
    @test counts(clu3, clu5) == counts(clu5, clu3)'
    @test size(counts(clu3, clu5)) == (3, 5)
    @test sum(counts(clu3, clu5)) == 25
    @test counts(clu3, clu5) == counts(clu3, assignments(clu5))
    @test counts(clu3, clu5) == counts(assignments(clu3), clu5)
    @test counts(clu3, clu5) == counts(assignments(clu3), assignments(clu5))
end

end
