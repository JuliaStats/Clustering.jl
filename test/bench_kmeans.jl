# a simple benchmark to compare with old-implementation

# new one is named kmeans, which considers each column as a sample
# old one is named k_means, which considers each row as a sample

using Clustering
using OptionsMod

m = 100
n = 10000
k = 50

x = rand(m, n)

println("running new k-means")
opts = @options display=:none max_iter=200
t_new = @elapsed r_new = kmeans(x, k, opts)
println("\telapsed = $t_new, per_iteration = $(t_new / r_new.iterations)")
println()

println("running old k-means")
t_old = @elapsed r_old = k_means(x, k)
println("\telapsed = $t_old, per_iteration = $(t_old / r_old.iterations)")

gain = (t_old / r_old.iterations) / (t_new / r_new.iterations)
println("gain = $gain")

