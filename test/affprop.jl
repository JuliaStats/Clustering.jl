# simple program to test affinity propagation

using Base.Test
using Distances
using Clustering

srand(34568)

d = 10
n = 500
x = rand(d, n)
S = -pairwise(Euclidean(), x, x)

# set diagonal value to median value
S = S - diagm(diag(S)) + median(S)*eye(size(S,1)) 

R = affinityprop(S)

@test isa(R, AffinityPropResult)
k = length(R.exemplars)
@test 0 < k < n
@test length(R.assignments) == n
@test all(R.assignments .>= 1) && all(R.assignments .<= k)
@test all(R.assignments[R.exemplars] .== [1:k])

@test length(R.counts) == k
@test sum(R.counts) == n
for i = 1:k
	@test R.counts[i] == countnz(R.assignments .== i)
end

