testfiles = ["affprop.jl",
            "dbscan_test.jl",
            "kmeans.jl",
            "kmedoids.jl",
            "sil.jl"]

for fname in testfiles
    println("Running $fname...")
    include(fname)
end

