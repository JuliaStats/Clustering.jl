using BenchmarkTools
using Clustering, Distances, Random

Random.seed!(345678)

const SUITE = BenchmarkGroup()

SUITE["hclust"] = BenchmarkGroup()

function random_distance_matrix(n::Integer, m::Integer=10, dist::PreMetric=Euclidean())
    pts = rand(m, n)
    return pairwise(dist, pts, dims=2)
end

function hclust_benchmark(n::Integer, m::Integer=10, dist::PreMetric=Euclidean())
    res = BenchmarkGroup()
    for linkage in ("single", "average", "complete")
        res[linkage] = @benchmarkable hclust(D, linkage=Symbol($linkage)) setup=(D=random_distance_matrix($n, $m, $dist))
    end
    return res
end

for n in (10, 100, 1000, 10000)
    SUITE["hclust"][n] = hclust_benchmark(n)
end

SUITE["cutree"] = BenchmarkGroup()

for (n, k) in ((10, 3), (100, 10), (1000, 100), (10000, 1000))
    SUITE["cutree"][(n,k)] = @benchmarkable cutree(hclu, k=$k) setup=(D=random_distance_matrix($n, 5); hclu=hclust(D, linkage=:single))
end

function silhouette_benchmark(metric, assgns, points, nclusters)
    res = BenchmarkGroup()
    res[:distances] = @benchmarkable silhouettes($assgns, pairwise($metric, $points, $points, dims=2))
    res[:points] = @benchmarkable silhouettes($assgns, $points; metric=$metric)
    return res
end

SUITE["silhouette"] = BenchmarkGroup()
for metric in [SqEuclidean(), Euclidean()]
    SUITE["silhouette"]["metric=$(typeof(metric))"] = metric_bench = BenchmarkGroup()
    for n in [100, 1000, 10000, 20000]
        nclusters = 10
        dims = 10
        points = randn(dims, n)
        assgns = rand(1:nclusters, n)
        metric_bench["n=$n"] = silhouette_benchmark(metric, assgns, points, nclusters)
    end
end