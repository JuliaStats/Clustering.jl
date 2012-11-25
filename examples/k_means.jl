load("Clustering")
using Clustering

srand(1)

n = 100

x = vcat(randn(n, 2), randn(n, 2) .+ 10)
true_assignments = vcat(zeros(n), ones(n))

results = k_means(x, 2)
results.assignments
