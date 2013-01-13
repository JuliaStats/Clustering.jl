using DataFrames, RDatasets, Clustering

iris = data("datasets", "iris")

k = length(unique(iris[:, "Species"]))

clusters = k_means(matrix(iris[:, 2:5]), k)

df = DataFrame()
df["Cluster"] = clusters.assignments
df["Label"] = iris[:, "Species"]
head(df)

# Clustering doesn't work well using some of the numeric features
# Let's try subsets of them to see what happens
df = DataFrame()
df["AllCluster"] = k_means(iris[:, 2:5], k).assignments
df["SepalCluster"] = k_means(iris[:, 2:3], k).assignments
df["PetalCluster"] = k_means(iris[:, 4:5], k).assignments
df["Label"] = iris[:, "Species"]

# Beyond the k-means algorithm we can also use dp-means
clusters = dp_means(iris[:, 2:5], 5.0)
