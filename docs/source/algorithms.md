# [Basics](@id clu_algo_basics)

The package implements a variety of clustering algorithms:

```@contents
Pages = ["kmeans.md", "kmedoids.md", "hclust.md", "mcl.md",
         "affprop.md", "dbscan.md", "fuzzycmeans.md"]
```

Most of the clustering functions in the package have a similar interface,
making it easy to switch between different clustering algorithms.

## Inputs

A clustering algorithm, depending on its nature, may accept an input
matrix in either of the following forms:

  - Data matrix ``X`` of size ``d \times n``, the ``i``-th column of ``X``
    (`X[:, i]`) is a data point (data *sample*) in ``d``-dimensional space.
  - Distance matrix ``D`` of size ``n \times n``, where ``D_{ij}`` is the
    distance between the ``i``-th and ``j``-th points, or the cost of assigning
    them to the same cluster.

## [Common Options](@id common_options)

Many clustering algorithms are iterative procedures. The functions share the
basic options for controlling the iterations:
 - `maxiter::Integer`: maximum number of iterations.
 - `tol::Real`: minimal allowed change of the objective during convergence.
   The algorithm is considered to be converged when the change of objective
   value between consecutive iterations drops below `tol`.
 - `display::Symbol`: the level of information to be displayed. It may take one
   of the following values:
   * `:none`: nothing is shown
   * `:final`: only shows a brief summary when the algorithm ends
   * `:iter`: shows the progress at each iteration

## Results

A clustering function would return an object (typically, an instance of
some [`ClusteringResult`](@ref) subtype) that contains both the resulting
clustering (e.g. assignments of points to the clusters) and the information
about the clustering algorithm (e.g. the number of iterations and whether it
converged).

```@docs
ClusteringResult
```

The following generic methods are supported by any subtype of `ClusteringResult`:
```@docs
nclusters(::ClusteringResult)
counts(::ClusteringResult)
wcounts(::ClusteringResult)
assignments(::ClusteringResult)
```
