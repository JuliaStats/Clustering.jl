using Clustering
using Test
using Random
using LinearAlgebra
using SparseArrays
if VERSION >= v"0.7.0-beta.85"
    using Statistics
end

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
