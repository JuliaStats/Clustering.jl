using Test
using Clustering, Distances

@testset "clustering_quality()" begin

    # test data with 12 2D points and 4 clusters
    Y = [-2 2 2 3  2 1  2 -2 -2 -1 -2 -3
          4 4 1 0 -1 0 -4 -4  1  0 -1  0]
    # cluster centers
    C = [0 2  0 -2
         4 0 -4  0]
    # point-to-cluster assignments
    A = [1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]
    # convert A to fuzzy clusters weights
    W = zeros(Int, (size(Y, 2), size(C, 2)))
    for (i, c) in enumerate(A)
        W[i, c] = 1
    end
    # fuzzy clustering with 4 clusters
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
    # mock hard and fuzzy clusterings for testing interface; only C, W and A arguments are actually used
    A_kmeans = KmeansResult(Float64.(C), A, ones(12), [4, 4, 4], ones(4), 42., 42, true)
    W_cmeans = FuzzyCMeansResult(Float64.(C), Float64.(W), 42, true)
    W2_cmeans = FuzzyCMeansResult(Float64.(C), Float64.(W2), 42, true)

    @testset "input checks" begin
        @test_throws ArgumentError clustering_quality(zeros(2,2), zeros(2,3), [1, 2], quality_index = :calinski_harabasz)
        @test_throws DimensionMismatch clustering_quality(zeros(2,2), zeros(3,2), [1, 2], quality_index = :calinski_harabasz)
        @test_throws ArgumentError clustering_quality(zeros(2,2), zeros(2,1), [1, ], quality_index = :calinski_harabasz)
        @test_throws ArgumentError clustering_quality(zeros(2,2), zeros(2,2), [1, 2], quality_index = :calinski_harabasz)
        @test_throws ArgumentError clustering_quality(zeros(0,0), zeros(0,0), zeros(Int,0); quality_index = :calinski_harabasz)
        @test_throws ArgumentError  clustering_quality(zeros(0,0), zeros(0,0), zeros(0,0); quality_index = :calinski_harabasz, fuzziness = 2)
        @test_throws DimensionMismatch clustering_quality([1,2,3], zeros(2,2), quality_index = :dunn)
        # wrong quality index
        @test_throws ArgumentError clustering_quality(Y, C, A; quality_index = :nonexistent_index)
        @test_throws ArgumentError clustering_quality(Y, C, W; quality_index = :nonexistent_index, fuzziness = 2)
        @test_throws ArgumentError clustering_quality(Y, A; quality_index = :nonexistent_index)
    end

    @testset "correct quality index values" begin
        @testset "calinski_harabasz" begin
            @test clustering_quality(Y, C, A; quality_index = :calinski_harabasz, metric = Euclidean()) ≈ (32/3) / (16/8)
            @test clustering_quality(Y, A_kmeans; quality_index = :calinski_harabasz, metric = Euclidean()) ≈ (32/3) / (16/8)
            # requires centers
            @test_throws ArgumentError clustering_quality(A_kmeans, pairwise(Euclidean(), Y, dims=2); quality_index = :calinski_harabasz)

            @test clustering_quality(Y, C, W; quality_index = :calinski_harabasz, fuzziness = 2, metric = Euclidean()) ≈ (32/3) / (16/8)
            @test clustering_quality(Y, W_cmeans; quality_index = :calinski_harabasz, fuzziness = 2, metric = Euclidean()) ≈ (32/3) / (16/8)
            @test_throws MethodError clustering_quality(W_cmeans, pairwise(Euclidean(), Y, dims=2); quality_index = :calinski_harabasz, fuzziness = 2) ≈ (32/3) / (16/8)

            @test clustering_quality(Y, C, W2; quality_index = :calinski_harabasz, fuzziness = 2, metric = Euclidean()) ≈ 8/3 * ( 24 ) / (14+sqrt(17))
            @test clustering_quality(Y, W2_cmeans; quality_index = :calinski_harabasz, fuzziness = 2, metric = Euclidean()) ≈ 8/3 * ( 24 ) / (14+sqrt(17))
            @test_throws MethodError clustering_quality(W2_cmeans, pairwise(Euclidean(), Y, dims=2); quality_index = :calinski_harabasz, fuzziness = 2)
        end

        @testset "davies_bouldin" begin
            @test clustering_quality(Y, C, A; quality_index = :davies_bouldin, metric = Euclidean()) ≈ 3/sqrt(20)
            @test clustering_quality(Y, A_kmeans; quality_index = :davies_bouldin, metric = Euclidean()) ≈ 3/sqrt(20)
            # requires centers
            @test_throws ArgumentError clustering_quality(A_kmeans, pairwise(Euclidean(), Y, dims=2); quality_index = :davies_bouldin) ≈ 3/sqrt(20)
            # fuzziness not supported
            @test_throws ArgumentError clustering_quality(Y, W_cmeans; quality_index = :davies_bouldin, fuzziness = 2)
        end

        @testset "dunn" begin
            @test clustering_quality(Y, C, A; quality_index = :dunn, metric = Euclidean()) ≈ 1/2
            @test clustering_quality(Y, A_kmeans; quality_index = :dunn, metric = Euclidean()) ≈ 1/2
            @test clustering_quality(A_kmeans, pairwise(Euclidean(), Y, dims=2); quality_index = :dunn) ≈ 1/2
            # fuzziness not supported
            @test_throws ArgumentError clustering_quality(Y, W_cmeans; quality_index = :dunn, fuzziness = 2)
        end

        @testset "xie_beni" begin
            @test clustering_quality(Y, C, A; quality_index = :xie_beni, metric = Euclidean()) ≈ 1/3

            @test clustering_quality(Y, C, W; quality_index = :xie_beni, fuzziness = 2, metric = Euclidean()) ≈ 1/3
            @test clustering_quality(Y, W_cmeans; quality_index = :xie_beni, fuzziness = 2, metric = Euclidean()) ≈ 1/3

            @test clustering_quality(Y, C, W2; quality_index = :xie_beni, fuzziness = 2, metric = Euclidean()) ≈ (14+sqrt(17)) / (12 * 4)
            @test clustering_quality(Y, W2_cmeans; quality_index = :xie_beni, fuzziness = 2, metric = Euclidean()) ≈ (14+sqrt(17)) / (12 * 4)
        end

        @testset "silhouettes" begin
            avg_silh = 1 - 1/12*( # average over silhouettes 1 - a_i * 1/b_i
                + 4 * 16 /(3+2sqrt(17)+5) # 4 points in clusters 1 and 3
                + 4 * (2sqrt(2)+2)/3 * 1/4 # 4 points in clusters 2 and 4, top + bottom
                + 2 * (2sqrt(2)+2)/3 * 4/(4+2sqrt(26)+6) # 2 points clusters 2 and 4, left and right
                + 2 * (2sqrt(2)+2)/3 * 4/(2+2sqrt(10)+4) # 2 points clusters 2 and 4, center
            )
            @test clustering_quality(Y, A; quality_index = :silhouettes, metric = Euclidean()) ≈ avg_silh
            @test clustering_quality(Y, A_kmeans; quality_index = :silhouettes, metric = Euclidean()) ≈ avg_silh
            @test clustering_quality(A_kmeans, pairwise(Euclidean(), Y, dims=2); quality_index = :silhouettes) ≈ avg_silh
            # fuzziness not supported
            @test_throws ArgumentError clustering_quality(Y, W_cmeans; quality_index = :silhouettes, fuzziness = 2)
        end
    end

    @testset "empty clusters" begin
        # degenerated clustering, no 4th cluster
        degenC = [0 2  0 -2 -2
                  4 0 -4  0  0]
        degenA = [1, 1, 2, 2, 2, 2, 3, 3, 5, 5, 5, 5]

        @test_logs((:warn, "Detected empty cluster(s): #4. clustering_quality() results might be incorrect."),
                   clustering_quality(Y, degenC, degenA; quality_index = :calinski_harabasz))
        @test clustering_quality(Y, degenC, degenA; quality_index = :calinski_harabasz, metric = Euclidean()) ≈ (32/3) / (16/8)
    end

end