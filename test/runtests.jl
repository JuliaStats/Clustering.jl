
tests = ["seeding",
         "kmeans",
         "kmedoids",
         "affprop",
         "dbscan",
         "silhouette", 
         "varinfo",
         "randindex",
         "hclust"]

println("Runing tests:")
for t in tests
    fp = "$(t).jl"
    println("* $fp ...")
    include(fp)
end
