using Test
using Clustering

D = [0 1 2 3
     1 0 1 2
     2 1 0 1
     3 2 1 0]

@assert size(D) == (4, 4)

a = [1, 1, 2, 2]
c = [2, 2]

@test all(isapprox.(silhouettes(a, c, D), [1.5/2.5, 0.5/1.5, 0.5/1.5, 1.5/2.5]))

a = [1, 2, 1, 2]
c = [2, 2]

@test all(isapprox.(silhouettes(a, c, D), [0.0, -0.5, -0.5, 0.0]))

a = [1, 1, 1, 2]
c = [3, 1]

@test all(isapprox.(silhouettes(a, c, D), [0.5, 0.5, -1/3, 0.0]))