# simple program to test the new k-means (not ready yet)

using Clustering

m = 3
n = 1000
k = 10

x = rand(m, n)

println("non-weighted")
r = kmeans(x, k; max_iter=50, display=:iter)
println()

println("weighted")
w = rand(n)
r = kmeans(x, k; max_iter=50, display=:iter, weights=w)
println()

println("works on Vector{Float32}")
@assert typeof(kmeans(float32(x), k; max_iter=50, display=:none)) <: Clustering.KmeansResult{Float32}
