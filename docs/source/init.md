# [Initialization](@id clu_algo_init)

A clustering algorithm usually requires initialization before it could be
started.

## Seeding

*Seeding* is a type of clustering initialization, which provides a few
*seeds* -- points from a data set that would serve as the initial cluster
centers (one for each cluster).

Each seeding algorithm implemented by *Clustering.jl* is a subtype of
`SeedingAlgorithm`:
```@docs
SeedingAlgorithm
initseeds!
initseeds_by_costs!
```
There are several seeding methods described in the literature. *Clustering.jl*
implements three popular ones:
```@docs
KmppAlg
KmCentralityAlg
RandSeedAlg
```
In practice, we have found that *Kmeans++* is the most effective choice.

For convenience, the package defines the two wrapper functions that accept
the short name of the seeding algorithm and the number of clusters and take
care of allocating `iseeds` and applying the proper `SeedingAlgorithm`:
```@docs
initseeds
initseeds_by_costs
```
