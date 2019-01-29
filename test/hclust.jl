using Clustering
using Test
using CodecZlib

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

@testset "R hclust() generated examples" begin
# load the examples
hclu_examples_filename = joinpath(@__DIR__, "data", "hclust_generated_examples.jl.gz")
Base.include_string(@__MODULE__,
                    open(io -> read(io, String), GzipDecompressorStream, hclu_examples_filename),
                    hclu_examples_filename)

# test to make sure many random examples match R's implementation
@testset "example #$i (linkage=:$(example["linkage"]), n=$(size(example["D"], 1)))" for
        (i, example) in enumerate(examples)

    linkage = example["linkage"]
    hclu = @inferred(hclust(example["D"], linkage=linkage))
    @test hclu isa Clustering.Hclust
    @test Clustering.nnodes(hclu) == size(example["D"], 1)
    @test Clustering.nmerges(hclu) == Clustering.nnodes(hclu)-1
    @test Clustering.height(hclu) ≈ maximum(example["height"]) atol=1e-5
    @test hclu.merges == example["merge"]
    @test hclu.heights ≈ example["height"] atol=1e-5
    @test hclu.order == example["order"]

    # compare hclust_nn_lw() (the default) and hclust_nn() (slower) methods
    if linkage ∈ [:complete, :average]
        @testset "hclust_nn()" begin
            hclu2 = Hclust(Clustering.hclust_nn(example["D"], linkage == :complete ? Clustering.slicemaximum : Clustering.slicemean),
                           linkage)
            @test hclu2.merges == hclu.merges
            @test hclu2.heights ≈ hclu.heights atol=1e-5
            @test hclu2.order == hclu.order
        end
    end

    local cut_k = example["cut_k"]
    local cut_h = example["cut_h"]
    if cut_h !== nothing
        # due to small arithmetic differences between R and Julia heights might be slightly different
        # find the matching height
        cut_h_r = cut_h
        cut_h_ix = findmin(abs.(hclu.heights .- cut_h_r))[2]
        cut_h = hclu.heights[cut_h_ix]
        @assert isapprox(cut_h, cut_h_r, atol=1e-6) "h=$cut_h ≈ h_R=$cut_h_r"
    end
    @testset "cutree(hclu, k=$(repr(cut_k)), h=$(repr(cut_h)))" begin
        cutt = @inferred(cutree(hclu, k=cut_k, h=cut_h))
        @test cutt isa Vector{Int}
        @test length(cutt) == Clustering.nnodes(hclu)
        @test cutt == example["cutree"]
    end
end
end

@testset "hclust_n3()" begin
    # no thorough testing (it's O(N³)), just test one example
    example_n3 = examples[10]
    hclu_n3 = @inferred(Clustering.hclust_n3(example_n3["D"], Clustering.slicemaximum))
    @test hclu_n3.mleft == example_n3["merge"][:, 1]
    @test hclu_n3.mright == example_n3["merge"][:, 2]
    @test hclu_n3.heights ≈ example_n3["height"] atol=1e-5
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

@testset "cutree(hclust, h=h) when the height of all subtrees greater than h (#141)" begin
    A = [0.0 0.7; 0.7 0.0]
    hA = hclust(A, linkage=:average)
    @test cutree(hA, h=0.5) == [1, 2]
    @test cutree(hA, h=0.7) == [1, 1]
    @test cutree(hA, h=0.9) == [1, 1]
end

end
