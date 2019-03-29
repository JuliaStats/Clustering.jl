# K-means

[K-means](http://en.wikipedia.org/wiki/K_means) is a classical method for
clustering or vector quantization. It produces a fixed number of clusters,
each associated with a *center* (also known as a *prototype*), and each data
point is assigned to a cluster with the nearest center.

From a mathematical standpoint, K-means is a coordinate descent
algorithm that solves the following optimization problem:
```math
\text{minimize} \ \sum_{i=1}^n \| \mathbf{x}_i - \boldsymbol{\mu}_{z_i} \|^2 \ \text{w.r.t.} \ (\boldsymbol{\mu}, z)
```
Here, ``\boldsymbol{\mu}_k`` is the center of the ``k``-th cluster, and
``z_i`` is an index of the cluster for ``i``-th point ``\mathbf{x}_i``.

```@docs
kmeans
KmeansResult
```

If you already have a set of initial center vectors, [`kmeans!`](@ref)
could be used:

```@docs
kmeans!
```

## Examples

```@example julia
using Clustering

# make a random dataset with 1000 random 5-dimensional points
X = rand(5, 1000)

# cluster X into 20 clusters using K-means
R = kmeans(X, 20; maxiter=200, display=:iter)

@assert nclusters(R) == 20 # verify the number of clusters

a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers
```

```@example Scatter plot of the K-means clustering results
using RDatasets, Clustering, Plots
iris = dataset("datasets", "iris"); # load the data

features = collect(Matrix(iris[:, 1:4])'); # features to use for clustering
result = kmeans(features, 3); # run K-means for the 3 clusters

# plot with the point color mapped to the assigned cluster index
scatter(iris.PetalLength, iris.PetalWidth, marker_z=result.assignments,
        color=:lightrainbow, legend=false)
```
