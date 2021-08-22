using Test
using Clustering

@testset "fmeasure()" begin


    a1 = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    a2 = [1, 1, 1, 1, 1, 2, 3, 3, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    @test fmeasure(a1, a2) ≈ 0.47 atol=1.0e-2
    @test pair_precision(a1, a2) ≈ 0.5 atol=1.0e-2
    @test pair_recall(a1, a2) ≈ 0.45 atol=1.0e-2


    a1 = [1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 2]
    a2 = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
    @test fmeasure(a1, a2) ≈ 0.529 atol=1.0e-2
    @test pair_precision(a1, a2) ≈ 0.6 atol=1.0e-2
    @test pair_recall(a1, a2) ≈ 0.47 atol=1.0e-2

end
