# simple program to test affinity propagation

using Distance
using Clustering

x = rand(100, 500)
S = -pairwise(Euclidean(), x, x)

# set diagonal value to median value
S = S - diagm(diag(S)) + median(S)*eye(size(S,1))

result = affinity_propagation(S; max_iter=200, display=:iter)
