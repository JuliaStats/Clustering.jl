# Clustering.jl

Methods for data clustering and evaluation of clustering quality.

[![Travis](https://travis-ci.org/JuliaStats/Clustering.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Clustering.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaStats/Clustering.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaStats/Clustering.jl?branch=master)

**Documentation**: [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

## Installation

```julia
Pkg.add("Clustering")
```

## Features

### Clustering Algorithms

- K-means
- K-medoids
- Affinity Propagation
- Density-based spatial clustering of applications with noise (DBSCAN)
- Markov Clustering Algorithm (MCL)
- Fuzzy C-Means Clustering
- Hierarchical Clustering
  - Single Linkage
  - Average Linkage
  - Complete Linkage
  - Ward's Linkage

### Clustering Validation

- Silhouettes
- Variation of Information
- Rand index
- V-Measure

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://JuliaStats.github.io/Clustering.jl/latest/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://JuliaStats.github.io/Clustering.jl/stable/

## See Also

Julia packages providing other clustering methods:
 - [QuickShiftClustering.jl](https://github.com/rened/QuickShiftClustering.jl)
 - [SpectralClustering.jl](https://github.com/lucianolorenti/SpectralClustering.jl)
