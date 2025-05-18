using Test
using Clustering

@testset "mutualinfo() (mutual information)" begin

    # https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    a1 = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    a2 = [1, 1, 1, 1, 1, 2, 3, 3, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    @test mutualinfo(a1, a2; method=:classic) ≈ 0.39 atol=1.0e-2
    @test mutualinfo(a1, a2;) ≈ 0.36 atol=1.0e-2
    @test mutualinfo(a1, a2; method=:adjusted) ≈ 0.2602 atol=1.0e-4
    @test mutualinfo(a1, a2; method=:adjusted, aggregate=:geomean) ≈ 0.2602 atol=1.0e-4
    @test mutualinfo(a1, a2; method=:adjusted, aggregate=:max) ≈ 0.2547 atol=1.0e-4
    @test mutualinfo(a1, a2; method=:adjusted, aggregate=:min) ≈ 0.2659 atol=1.0e-4

    # test deprecated kwarg
    @test mutualinfo(a1, a2; normed=false) ≈ 0.39 atol=1.0e-2
    @test mutualinfo(a1, a2; normed=true) ≈ 0.36 atol=1.0e-2

    # https://doi.org/10.1186/1471-2105-7-380
    a1 = [1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 2]
    a2 = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
    @test mutualinfo(a1, a2; method=:classic) ≈ 0.6 atol=0.1
    @test mutualinfo(a1, a2; method=:normalized) ≈ 0.5 atol=0.1
    @test mutualinfo(a1, a2; method=:adjusted, aggregate=:mean) ≈ 0.3839 atol=1.0e-4
    @test mutualinfo(a1, a2; method=:adjusted, aggregate=:geomean) ≈ 0.3861 atol=1.0e-4
    @test mutualinfo(a1, a2; method=:adjusted, aggregate=:max) ≈ 0.3437 atol=1.0e-4
    @test mutualinfo(a1, a2; method=:adjusted, aggregate=:min) ≈ 0.4348 atol=1.0e-4

    # test errors. More precise tests on Julia 1.8+ when supported

    if VERSION >= v"1.8"
        @test_throws "ArgumentError: `normed` kwarg is not compatible with `method` kwarg" mutualinfo(a1, a2; method=:adjusted, normed=false)
        @test_throws "ArgumentError: mutualinfo(): `method=:adjusfted` is not supported" mutualinfo(a1, a2; method=:adjusfted, aggregate=:min)
        @test_throws "ArgumentError: mutualinfo(): unsupported kwargs used." mutualinfo(a1, a2; method=:adjusted, notaggregate=:min)
        @test_throws "ArgumentError: mutualinfo(): unsupported kwargs used." mutualinfo(a1, a2; method=:classic, notaggregate=:min)
    else
        @test_throws ArgumentError mutualinfo(a1, a2; method=:adjusted, normed=false)
        @test_throws ArgumentError mutualinfo(a1, a2; method=:adjusfted, aggregate=:min)
        @test_throws ArgumentError mutualinfo(a1, a2; method=:adjusted, notaggregate=:min)
        @test_throws ArgumentError mutualinfo(a1, a2; method=:classic, notaggregate=:min)
    end
end
