
tests = ["seeding",
         "kmeans",
         "dpmeans",
         "kmedoids",
         "affprop",
         "dbscan",
         "silhouette", 
         "varinfo"]

println("Runing tests:")
for t in tests
    fp = "$(t).jl"
    println("* $fp ...")
    include(fp)
end

