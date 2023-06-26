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

SUITE["silhouette"] = BenchmarkGroup()

precalc_val(pre_calculate) = pre_calculate == "without_precalculation" ? false : true
function silhouette_benchmark(a, X, nclusters)
    res = BenchmarkGroup()
    for pre_calculate in ("without_precalculation", "with_precalculation")
        res[pre_calculate] = @benchmarkable silhouettes(SqEuclidean(), $a, $X; nclusters=$nclusters, pre_calculate=precalc_val($pre_calculate))
    end
    return res
end

for (label, n) in (("n=100", 100), ("n=1,000", 1000), ("n=10,000", 10000), ("n=20,000", 20000))
    nclusters = 10
    dims = 3
    X = rand(dims, n); a = rand(1:nclusters, n)
    SUITE["silhouette"][label] = silhouette_benchmark(a, X, nclusters)
end