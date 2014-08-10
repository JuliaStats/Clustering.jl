
tests = ["seeding",
         "kmeans",
         "kmedoids",
         "affprop",
         "dbscan",
         "silhouette"]

println("Runing tests:")
for t in tests
    fp = "$(t).jl"
    println("* $fp ...")
    include(fp)
end

