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
    hclu_n3 = Clustering.hclust_n3(example_n3["D"], maximum)
    @test hclu_n3[1] == example_n3["merge"]
    @test hclu_n3[2] ≈ example_n3["height"] atol=1e-5
end

end
