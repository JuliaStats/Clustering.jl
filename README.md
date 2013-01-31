# Installation

    Pkg.add("Clustering")

# Functionality

* k_means
* dp_means

# Examples

    using DataFrames, RDatasets, Clustering

    iris = data("datasets", "iris")

    k = length(unique(iris[:, "Species"]))

    clusters = k_means(matrix(iris[:, 2:5]), k)

    df = DataFrame()
    df["Cluster"] = clusters.assignments
    df["Label"] = iris[:, "Species"]
    head(df)

    by(df, ["Cluster", "Label"], nrow)

    clusters = dp_means(matrix(iris[:, 2:5]), 6.0)

    df = DataFrame()
    df["Cluster"] = clusters.assignments
    df["Label"] = iris[:, "Species"]
    head(df)

    by(df, ["Cluster", "Label"], nrow)
