using Clustering
using Test
using CodecZlib
using Distances

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
    Random.seed!(1)
    D = rand(50,50)
    Dm = D + D'
    hcl_rand = hclust(Dm, linkage=:average, branchorder=:optimal)

    @test hcl_rand.merges == [-29 -1; -32 -24; -46 -44; -10 -41; -17 -12; -40 5;
                              -8 -28; -13 -35; -19 -20; -43 -42; -34 -18; -15 -39;
                              -30 -49; -26 -22; -36 -31; -38 -4; -5 -9; 1 16; -6 13;
                              17 10; -33 -45; -7 -2; -23 -21; -48 -27; -37 -16; -14 8;
                              2 -50; 19 20; -47 -11; 9 -3; 6 15; 14 22; 31 18; 23 25;
                              11 3; 32 29; 33 30; 7 12; 35 38; -25 4; 21 28; 26 34;
                              41 27; 24 40; 39 42; 37 43; 45 46; 47 36; 48 44]

    @test hcl_rand.order == [34, 18, 46, 44, 8, 28, 15, 39, 14, 13, 35, 23, 21, 37,
                             16, 40, 17, 12, 36, 31, 29, 1, 38, 4, 19, 20, 3, 33, 45,
                             6, 30, 49, 5, 9, 43, 42, 32, 24, 50, 26, 22, 7, 2, 47,
                             11, 48, 27, 25, 10, 41]

    @test Clustering.nnodes(hcl_rand) == 50
end

end # testset "hclust()"
