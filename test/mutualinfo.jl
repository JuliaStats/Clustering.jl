using Test
using Clustering

@testset "mutualinfo() (mutual information)" begin

    # https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    a1 = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    a2 = [1, 1, 1, 1, 1, 2, 3, 3, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    @test mutualinfo(a1, a2, normed=false) ≈ 0.39 atol=1.0e-2
    @test mutualinfo(a1, a2) ≈ 0.36 atol=1.0e-2

    # https://doi.org/10.1186/1471-2105-7-380
    a1 = [1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 2]
    a2 = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
    @test mutualinfo(a1, a2, normed=false) ≈ 0.6 atol=0.1
    @test mutualinfo(a1, a2) ≈ 0.5 atol=0.1

end
