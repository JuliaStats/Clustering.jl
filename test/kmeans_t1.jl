# simple program to test the new k-means (not ready yet)

using Clustering
using OptionsMod
using Distance

m = 3
n = 1000
k = 10

x = rand(m, n)

println("non-weighted")
opts = @options max_iter=50 display=:iter
r = kmeans(x, k, opts)
println()

println("weighted")
w = rand(n)
opts = @options max_iter=50 display=:iter weights=w
r = kmeans(x, k, opts)
println()