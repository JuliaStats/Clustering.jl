using Plots, Clustering, Distances, Statistics

X = hcat([4., 5.] .+ 0.2 * randn(2, 10),
         [9., -5.] .+ 0.2 * randn(2, 5),
         [-4., -9.] .+ 0.5 * randn(2, 5))


scatter(X[1,:],X[2,:],
    label = nothing,
)

##
resf = fuzzy_cmeans(X,3,2)

res = kmeans(X,3)

q = [calinski_harabasz(X,kmeans(X,k)) for k in 2:5]
q = [xie_beni(X,kmeans(X,k)) for k in 2:5]
q = [davies_bouldin(X,kmeans(X,k)) for k in 2:5]
q = [dunn(X,kmeans(X,k),SqEuclidean()) for k in 2:5]

q = [calinski_harabasz(X,fuzzy_cmeans(X,k,2), 2) for k in 2:5]
q = [xie_beni(X,fuzzy_cmeans(X,k,2), 2) for k in 2:5]

plot(2:5,q)

## test data

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
scatter(Y[:,1],Y[:,2],
    axisratio = :equal,
    #seriescolor = palette(default)[A],
)
scatter!(C[:,1],C[:,2],
    marker = :square
)

## tests
using Test

@test_throws ArgumentError Clustering._check_qualityindex_arguments(zeros(2,2), zeros(2,3), [1, 2])
@test_throws DimensionMismatch Clustering._check_qualityindex_arguments(zeros(2,2),zeros(3,2), [1, 2])
@test_throws ArgumentError Clustering._check_qualityindex_arguments(zeros(2,2),zeros(2,1), [1, ])
@test_throws ArgumentError Clustering._check_qualityindex_arguments(zeros(2,2),zeros(2,2), [1, 2])

@test calinski_harabasz(Y',C',A,Euclidean()) ≈ (32/3) / (16/8)
@test calinski_harabasz(Y',C',W,2,Euclidean()) ≈ (32/3) / (16/8)

@test davies_bouldin(Y',C',A,Euclidean()) ≈ 3/2√5

@test xie_beni(Y',C',A,Euclidean()) ≈ 1/3
@test xie_beni(Y',C',W,2,Euclidean()) ≈ 1/3

@test dunn(Y',A,Euclidean()) ≈ 1/2