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
    W2 = [
        1  0  0  0
        1  0  0  0
        0 1/2 0 1/2
        0 1/2 0 1/2
        0 1/2 0 1/2
        0 1/2 0 1/2
        0  0  1  0
        0  0  1  0
        0 1/2 0 1/2
        0 1/2 0 1/2
        0 1/2 0 1/2
        0 1/2 0 1/2
    ]

    @testset "input checks" begin
        @test_throws ArgumentError clustering_quality(zeros(2,2), zeros(2,3), [1, 2], quality_index = :calinski_harabasz)
        @test_throws DimensionMismatch clustering_quality(zeros(2,2),zeros(3,2), [1, 2], quality_index = :calinski_harabasz)
        @test_throws ArgumentError clustering_quality(zeros(2,2),zeros(2,1), [1, ], quality_index = :calinski_harabasz)
        @test_throws ArgumentError clustering_quality(zeros(2,2),zeros(2,2), [1, 2], quality_index = :calinski_harabasz)
        @test_throws ArgumentError clustering_quality(zeros(0,0),zeros(0,0), zeros(Int,0); quality_index = :calinski_harabasz)
        @test_throws ArgumentError  clustering_quality(zeros(0,0), zeros(0,0),zeros(0,0); quality_index = :calinski_harabasz, fuzziness = 2)
        @test_throws DimensionMismatch clustering_quality([1,2,3], zeros(2,2), quality_index = :dunn)
        @test_throws ArgumentError clustering_quality(Y', C', A; quality_index = :nonexistent_index)
        @test_throws ArgumentError clustering_quality(Y', C', W; quality_index = :nonexistent_index, fuzziness = 2)
        @test_throws ArgumentError clustering_quality(Y', A; quality_index = :nonexistent_index)
    end

    @testset "correct index values" begin
        @test clustering_quality(Y', C', A; quality_index = :calinski_harabasz, metric = Euclidean()) ≈ (32/3) / (16/8)
        @test clustering_quality(Y', C', W; quality_index = :calinski_harabasz, fuzziness = 2, metric = Euclidean()) ≈ (32/3) / (16/8)
        @test clustering_quality(Y', C', W2; quality_index = :calinski_harabasz, fuzziness = 2, metric = Euclidean()) ≈ 8/3 * ( 24 ) / (14+sqrt(17))

        @test clustering_quality(Y', C', A; quality_index = :davies_bouldin, metric = Euclidean()) ≈ 3/sqrt(20)

        @test clustering_quality(Y', C', A; quality_index = :xie_beni, metric = Euclidean()) ≈ 1/3
        @test clustering_quality(Y', C', W; quality_index = :xie_beni, fuzziness = 2, metric = Euclidean()) ≈ 1/3
        @test clustering_quality(Y', C', W2; quality_index = :xie_beni, fuzziness = 2, metric = Euclidean()) ≈ (14+sqrt(17)) / (12 * 4)
        
        @test clustering_quality(Y', A; quality_index = :dunn, metric = Euclidean()) ≈ 1/2
    end

    @testset "alternate arguments" begin
        # mock hard and fuzzy clusterings for testing interface; only C, W and A arguments are actually used
        hardClustering = KmeansResult(Float64.(C'), A, ones(12), [4, 4, 4], ones(4), 42., 42, true)
        fuzzyClustering = FuzzyCMeansResult(Float64.(C'), Float64.(W), 42, true)
    
        @test clustering_quality(Y', hardClustering; quality_index = :calinski_harabasz, metric = Euclidean()) ≈ (32/3) / (16/8)
        @test clustering_quality(Y', fuzzyClustering; quality_index = :xie_beni, fuzziness = 2, metric = Euclidean()) ≈ 1/3
        @test clustering_quality(hardClustering, pairwise(Euclidean(), Y', dims=2); quality_index = :dunn) ≈ 1/2
    end

    @testset "empty clusters" begin
        # degenerated clustering
        degC = [0 4; 2 0; 0 -4; -2 0; -2 0]
        degA = [1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 5, 5] # no 4th cluster

        @test_logs (:warn, "Detected empty cluster(s) no.: 4. clustering_quality() results might be incorrect.") clustering_quality(Y', degC', degA; quality_index = :calinski_harabasz) 
    end

end