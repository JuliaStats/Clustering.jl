using Test
using Clustering, Distances

@testset "clustering_quality()" begin

    # test data with 4 clusters

    Y = [-2 4; 2 4; 2 1; 3 0; 2 -1; 1 0; 2 -4; -2 -4; -2 1; -1 0; -2 -1; -3 0]
    C = [0 4; 2 0; 0 -4; -2 0]
    A = [1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
    W = [
        1 0 0 0
        1 0 0 0
        0 1 0 0
        0 1 0 0
        0 1 0 0
        0 1 0 0
        0 0 1 0
        0 0 1 0
        0 0 0 1
        0 0 0 1
        0 0 0 1
        0 0 0 1
    ]

    # visualisation of the data
    # using Plots
    # scatter(Y[:,1],Y[:,2],
    #     axisratio = :equal,
    #     #seriescolor = palette(default)[A],
    # )
    # scatter!(C[:,1],C[:,2],
    #     marker = :square,
    #     label = "cluster centers",  
    # )

    @testset "input checks" begin
        @test_throws ArgumentError clustering_quality(zeros(2,2), zeros(2,3), [1, 2], quality_index = :calinski_harabasz)
        @test_throws DimensionMismatch clustering_quality(zeros(2,2),zeros(3,2), [1, 2], quality_index = :calinski_harabasz)
        @test_throws ArgumentError clustering_quality(zeros(2,2),zeros(2,1), [1, ], quality_index = :calinski_harabasz)
        @test_throws ArgumentError clustering_quality(zeros(2,2),zeros(2,2), [1, 2], quality_index = :calinski_harabasz)
        @test_throws DimensionMismatch clustering_quality([1,2,3], zeros(2,2), quality_index = :dunn)
    end

    @testset "correct index values" begin
        @test clustering_quality(Y', C', A; quality_index = :calinski_harabasz, metric = Euclidean()) ≈ (32/3) / (16/8)
        @test clustering_quality(Y', C', W; quality_index = :calinski_harabasz, fuzziness = 2, metric = Euclidean()) ≈ (32/3) / (16/8)

        @test clustering_quality(Y', C', A; quality_index = :davies_bouldin, metric = Euclidean()) ≈ 3/2 sqrt(5)

        @test clustering_quality(Y', C', A; quality_index = :xie_beni, metric = Euclidean()) ≈ 1/3
        @test clustering_quality(Y', C', W; quality_index = :xie_beni, fuzziness = 2, metric = Euclidean()) ≈ 1/3

        @test clustering_quality(Y', A; quality_index = :dunn, metric = Euclidean()) ≈ 1/2
    end

end