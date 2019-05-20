# Cross tabulation

[Cross tabulation](https://en.wikipedia.org/wiki/Contingency_table), or
*contingency matrix*, is a basis for many clustering quality measures.
It shows how similar are the two clusterings on a cluster level.

*Clustering.jl* extends `StatsBase.counts()` with methods that accept
[`ClusteringResult`](@ref) arguments:
```@docs
counts(a::ClusteringResult, b::ClusteringResult)
```
