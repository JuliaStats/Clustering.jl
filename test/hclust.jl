using Clustering
using Test
using CodecZlib
using Distances
using DelimitedFiles
using Random, StableRNGs

@testset "hclust() (hierarchical clustering)" begin

rng = StableRNG(42)

@testset "param checks" begin
    Random.seed!(rng, 42)
    D = rand(rng, 5, 5)
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
            linkage_fun = linkage == :complete ? Clustering.slicemaximum : Clustering.slicemean
            hclu2 = Hclust(Clustering.orderbranches_r!(Clustering.hclust_nn(example["D"], linkage_fun)),
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
    hclu_n3 = @inferred(Clustering.orderbranches_r!(Clustering.hclust_n3(example_n3["D"], Clustering.slicemaximum)))
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

@testset "Leaf ordering methods" begin
    n = 10
    mat = zeros(Int, n, n)

    for i in 1:n
        last = minimum([i+Int(floor(n/5)), n])
        for j in i:last
            mat[i,j] = 1
        end
    end

    dm = pairwise(Euclidean(), mat, dims=2)

    hcl_r = hclust(dm, linkage=:average)
    hcl_barjoseph = hclust(dm, linkage=:average, branchorder=:barjoseph)
    hcl_optimal = hclust(dm, linkage=:average, branchorder=:optimal)

    @test hcl_r.order == [3, 1, 2, 4, 5, 9, 10, 6, 7, 8]
    @test hcl_r.merges == [-1 -2; -3 1; -4 -5; -9 -10; -7 -8; -6 5; 2 3; 4 6; 7 8]

    @test hcl_barjoseph.order == collect(1:10)
    @test hcl_barjoseph.merges == [-1 -2; 1 -3; -4 -5; -9 -10; -7 -8; -6 5; 2 3; 6 4; 7 8]

    @test hcl_barjoseph.merges == hcl_optimal.merges
    @test hcl_barjoseph.order == hcl_optimal.order

    @test_throws ArgumentError hclust(dm, linkage=:average, branchorder=:wrong)

    hcl_zero = hclust(fill(0.0, 0, 0), linkage=:average, branchorder=:barjoseph)
    @test Clustering.nnodes(hcl_zero) == 0

    hcl_one = hclust(fill(0.0, 1, 1), linkage=:average, branchorder=:barjoseph)
    @test Clustering.nnodes(hcl_one) == 1

    # Larger matrix to make sure all swaps are tested
    Random.seed!(rng, 42)
    D = rand(rng, 50,50)
    Dm = D + D'
    hcl_rand = hclust(Dm, linkage=:average, branchorder=:optimal)

    merges = [-1 -36; -35 -34; 1 -16; -12 -41; -13 -30; -43 -24; -21 -32; -5 -18;
              -47 -22; -33 -38; -11 -2; -44 -49; -42 -10; 2 -37; -46 -25; -9 -15;
              -27 -8; -40 -39; -28 -19; -31 -6; 4 -14; 11 -50; -48 -4; 15 3; -23 -7;
              -20 -3; 18 -45; 5 -29; 9 6; 25 22; 21 7; -26 27; 8 10; 31 19; 13 14;
              34 24; 12 20; 30 33; 17 -17; 23 26; 16 37; 32 39; 29 35; 38 36; 41 28;
              45 43; 46 44; 42 47; 48 40]
    @test hcl_rand.merges == merges

    order = [26, 40, 39, 45, 27, 8, 17, 9, 15, 44, 49, 31, 6, 13, 30, 29, 47, 22,
             43, 24, 42, 10, 35, 34, 37, 23, 7, 11, 2, 50, 5, 18, 33, 38, 12, 41,
             14, 21, 32, 28, 19, 46, 25, 1, 36, 16, 48, 4, 20, 3]
    @test hcl_rand.order == order

    @test Clustering.nnodes(hcl_rand) == 50
end

@testset "Tree construction with duplicate distances (#176)" begin
    hclupi = hclust(fill(3.141592653589, 4, 4), linkage=:average)
    @test hclupi.heights == fill(3.141592653589, 3)
    @test hclupi.merges == [-1 -2; -4 1; -3 2]

    # check that the tree construction with the given matrix does not fail
    dist1_mtx = readdlm(joinpath(@__DIR__, "data", "hclust_dist_issue176_1.txt"), ',', Float64)
    hclu1_avg = hclust(dist1_mtx, linkage=:average)
    hclu1_min = hclust(dist1_mtx, linkage=:single)
    hclu1_ward = hclust(dist1_mtx, linkage=:ward)

    dist2_mtx = readdlm(joinpath(@__DIR__, "data", "hclust_dist_issue176_2.txt"), ',', Float64)
    hclu2_avg = hclust(dist2_mtx, linkage=:average)
    hclu2_min = hclust(dist2_mtx, linkage=:single)
    hclu2_ward = hclust(dist2_mtx, linkage=:ward)
end

@testset "cuttree() with merges not sorted by height (#252)" begin
    dist_mtx = readdlm(joinpath(@__DIR__, "data", "hclust_dist_issue252.txt"), ',', Float64)

    hclu_opt = hclust(dist_mtx, linkage=:complete, branchorder=:optimal)
    clu_opt = cutree(hclu_opt, h=20)

    hclu_r = hclust(dist_mtx, linkage=:complete)
    clu_r = cutree(hclu_r, h=20)

    @test clu_opt == clu_r
end

end # testset "hclust()"
