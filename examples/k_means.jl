load("Clustering")
using Clustering

srand(1)

n = 100

x = vcat(hcat(randn(n), randn(n)),
         hcat(randn(n) + 10, randn(n) + 10))
y = vcat(zeros(n), ones(n))

k_means(x, 2)
