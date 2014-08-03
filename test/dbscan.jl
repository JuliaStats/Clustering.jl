using Base.Test
using Clustering
using Distance

srand(34568)

X1 = randn(2, 200) .+ [0., 5.]
X2 = randn(2, 200) .+ [-5., 0.]
X3 = randn(2, 200) .+ [5., 0.]
X = hcat(X1, X2, X3)
n = size(X,2)

D = pairwise(Euclidean(), X)

R = dbscan(D, 1.0, 10)
@test isa(R, DbscanResult)
k = length(R.seeds)
# println("k = $k")
@test k == 3
@test all(R.assignments .<= k)
@test length(R.assignments) == n
@test length(R.counts) == k
for c = 1:k
    @test countnz(R.assignments .== c) == R.counts[c]
end
@test all(R.counts .>= 180)

