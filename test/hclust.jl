using Clustering
using Test

@testset "hclust() (hierarchical clustering)" begin

@testset "param checks" begin
    D = rand(5, 5)
    Dsym = D + D'
    Dnan = copy(Dsym)
    Dnan[1, 3] = Dnan[3, 1] = NaN
    @testset "hclust()" begin
        @test_throws ArgumentError hclust(Dsym, linkage=:typo)
        @test_throws ArgumentError hclust(D, linkage=:single)
        @test_throws ArgumentError hclust(Dnan, linkage=:single)
    end
    hclu = @inferred(hclust(Dsym, linkage=:single))
    @test hclu isa Clustering.Hclust
    @testset "cutree()" begin
        @test_throws ArgumentError cutree(hclu)
        @test_throws ArgumentError cutree(hclu, k=-1)
        @test_throws ArgumentError cutree(hclu, k=0)
        @test cutree(hclust(Dsym), k=10) isa Vector{Int}
        @test cutree(hclust(fill(0.0, 0, 0)), k=0) == Int[]
        @test cutree(hclust(fill(0.0, 0, 0)), k=1) == Int[]
    end
end

# load the examples array
include("hclust-generated-examples.jl")

# test to make sure many random examples match R's implementation
@testset "example #$i" for (i, example) in enumerate(examples)
    h = hclust(example["D"], linkage=example["method"])
    @test Clustering.nnodes(h) == size(example["D"], 1)
    @test Clustering.nmerges(h) == Clustering.nnodes(h)-1
    @test h.merges == example["merge"]
    @test h.heights ≈ example["height"] atol=1e-5
    @test h.order == example["order"]

    @testset "cutree()" begin
        # FIXME compare with R cuttree() result
        cutn2 = cutree(h, k=2)
        @test cutn2 isa Vector{Int}
        @test length(cutn2) == length(h.order)
        @test all(cl -> 1 <= cl <= 2, cutn2)
    end
end

@testset "hclust_n3()" begin
    # no thorough testing (it's O(N³)), just test one example
    example_n3 = examples[10]
    hclu_n3 = Clustering.hclust_n3(example_n3["D"], Clustering.slicemaximum)
    @test hclu_n3[1] == example_n3["merge"]
    @test hclu_n3[2] ≈ example_n3["height"] atol=1e-5
end

local hclust_linkages = [:single, :average, :complete]

@testset "hclust(0×0 matrix, linkage=$linkage)" for linkage in hclust_linkages
    hclu = hclust(fill(0.0, 0, 0), linkage=linkage)
    @test Clustering.nnodes(hclu) == 0
    @test Clustering.nmerges(hclu) == 0
    @test Clustering.height(hclu) == -Inf
    cut1 = @inferred(cutree(hclu, h=1))
    @test cut1 == Int[]
end

@testset "hclust([$dist] 1×1 matrix, linkage=$linkage)" for
    linkage in hclust_linkages, dist in [-Inf, 0, 1, 2, Inf]
    hclu = hclust(fill(dist, 1, 1), linkage=linkage)
    @test Clustering.nnodes(hclu) == 1
    @test Clustering.nmerges(hclu) == 0
    @test Clustering.height(hclu) == -Inf
    cut1 = @inferred(cutree(hclu, h=1))
    @test cut1 == [1]
end

@testset "hclust(linkage=$linkage) when data contains an isolated point (#109)" for linkage in hclust_linkages
    # point #2 is isolated: distances to all the other points are Inf
    mdist = [
        0.0 Inf 0.1  Inf  Inf  Inf  Inf  Inf  Inf  Inf;
        Inf 0.0 Inf  Inf  Inf  Inf  Inf  Inf  Inf  Inf;
        0.1 Inf 0.0  0.11 Inf  Inf  Inf  Inf  Inf  Inf;
        Inf Inf 0.11 0.0  0.86 0.86 Inf  Inf  Inf  Inf;
        Inf Inf Inf  0.86 0.0  0.67 0.72 1.93 Inf  Inf;
        Inf Inf Inf  0.86 0.67 0.0  0.72 Inf  Inf  Inf;
        Inf Inf Inf  Inf  0.72 0.72 0.0  Inf  0.72 Inf;
        Inf Inf Inf  Inf  1.93 Inf  Inf  0.0  3.91 Inf;
        Inf Inf Inf  Inf  Inf  Inf  0.72 3.91 0.0  0.52;
        Inf Inf Inf  Inf  Inf  Inf  Inf  Inf  0.52 0.0]

    hclu = hclust(mdist, linkage=linkage)
    @test Clustering.nnodes(hclu) == 10
    @test Clustering.nmerges(hclu) == 9
    @test Clustering.height(hclu) == Inf
    cut1 = @inferred(cutree(hclu, h=1))
    @test cut1 isa Vector{Int}
    @test length(cut1) == Clustering.nnodes(hclu)
    if linkage == :single
        @test cut1 == [1, 2, 1, 1, 1, 1, 1, 3, 1, 1]
    end
end

end
