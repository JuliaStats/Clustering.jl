using Clustering
using Test
using Random
using LinearAlgebra
using SparseArrays
using Statistics

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
         "vmeasure"]

println("Runing tests:")
for t in tests
    fp = "$(t).jl"
    println("* $fp ...")
    include(fp)
end
