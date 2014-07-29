
tests = ["seeding",
         "kmeans",
         "kmedoids",
         "affprop",
         "dbscan",
         "sil"]

for t in tests
    fp = "$(t).jl"
    println("Running $fp ...")
    include(fp)
end

