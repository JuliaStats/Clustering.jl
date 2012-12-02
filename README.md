# Installation

    require("pkg")
    Pkg.add("Clustering")

# Functionality

* k_means
* dp_means

# Examples

    load("DataFrames")
    using DataFrames

    load("RDatasets")
    using RDatasets

    load("Clustering")
    using Clustering

    iris = data("datasets", "iris")

    k = length(unique(iris[:, "Species"]))

    clusters = k_means(iris[:, 2:5], k)

    df = DataFrame()
    df["Cluster"] = clusters.assignments
    df["Label"] = iris[:, "Species"]
    head(df)
