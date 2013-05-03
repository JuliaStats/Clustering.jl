# a simple benchmark 

using Clustering

m = 100
n = 10000
k = 50

x = rand(m, n)

# warming
kmeans(x[:,1:1000], 2; display=:none, max_iter=200)

println("kmeans on $n samples (of dimension $m) with K = $k ...")
t_new = @elapsed r_new = kmeans(x, k; display=:none, max_iter=200)
println("\telapsed = $t_new, per_iteration = $(t_new / r_new.iterations)")
println()

