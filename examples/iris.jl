load("DataFrames")
using DataFrames

load("RDatasets")
using RDatasets

load("Clustering")
using Clustering

iris = data("datasets", "iris")

k = length(unique(iris[:, "Species"]))

clusters = k_means(matrix(iris[:, 2:5]), k)

df = DataFrame()
df["Cluster"] = clusters.assignments
df["Label"] = iris[:, "Species"]
head(df)

# * Clustering doesn't work well using some of the numeric features
# * Let's try subsets of them to see what happens
# * We'll use an alternative DataFrame syntax as well
#   an alternative API for k_means based on DataFrame inputs
DataFrame(quote
  AllCluster = k_means(iris[:, 2:5], k).assignments
  SepalCluster = k_means(iris[:, 2:3], k).assignments
  PetalCluster = k_means(iris[:, 4:5], k).assignments
  Label = iris[:, "Species"]
end)

# Beyond the k-means algorithm we can also use dp-means
clusters = dp_means(matrix(iris[:, 2:5]), 10.0)
clusters = dp_means(iris[:, 2:5], 10.0)
