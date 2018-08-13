using Clustering
using Test

@testset "hclust() (hierarchical clustering)" begin

# load the examples array
include("hclust-generated-examples.jl")

# test to make sure many random examples match R's implementation
@testset "example #$i" for (i, example) in enumerate(examples)
    h = hclust(example["D"], example["method"])
    @test h.merge == example["merge"]
    @test h.height ≈ example["height"] atol=1e-5
    @test h.order == example["order"]

    @testset "cutree()" begin
        # FIXME compare with R cuttree() result
        cutn2 = cutree(h, k=2)
        @test cutn2 isa Vector{Int}
        @test length(cutn2) == length(h.order)
        @test all(cl -> 1 <= cl <= 2, cutn2)
    end
end

end
