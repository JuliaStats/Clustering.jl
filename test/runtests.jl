include("../src/Clustering.jl")
using Compat
using Base.Test

tests = ["seeding",
         "kmeans",
         "kmedoids",
         "affprop",
         "dbscan",
         "fuzzycmeans",
         "silhouette",
         "varinfo",
         "randindex",
         "hclust",
         "mcl",
         "chinesewhispers"
        ]

println("Runing tests:")
for t in tests
    fp = "$(t).jl"
    @testset "$t" begin
        include(fp)
    end
end
