include("../src/Clustering.jl")
using Compat

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
         "mcl"]

println("Runing tests:")
for t in tests
    fp = "$(t).jl"
    println("* $fp ...")
    include(fp)
end
