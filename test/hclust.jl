using Clustering
using Test

@testset "hclust() (hierarchical clustering)" begin

# load the examples array
include("hclust-generated-examples.jl")

# test to make sure many random examples match R's implementation
@testset "example \"$example\"" for example in examples
    h = hclust(example["D"], example["method"])
    @test h.merge == example["merge"]
    @test h.height ≈ example["height"] atol=1e-5
    @test h.order == example["order"]
end

end
