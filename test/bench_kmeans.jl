# a simple benchmark to compare with old-implementation

# new one is named kmeans, which considers each column as a sample
# old one is named k_means, which considers each row as a sample

using Clustering
using OptionsMod

m = 100
n = 10000
k = 50

x = rand(m, n)

opts = @options display=:none max_iter=200

# warming
kmeans(x[:,1:1000], 2, opts)

println("kmeans on $n samples (of dimension $m) with K = $k ...")
t_new = @elapsed r_new = kmeans(x, k, opts)
println("\telapsed = $t_new, per_iteration = $(t_new / r_new.iterations)")
println()

