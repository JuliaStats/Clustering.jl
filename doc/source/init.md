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

For convenience, the package also defines the two wrapper methods that take
the name of the seeding algorithm and the number of clusters and take care of
allocating `iseeds` and applying the proper `SeedingAlgorithm` instance:
```@docs
initseeds
initseeds_by_costs
```

In practice, we found that *Kmeans++* is the most effective seeding method. To
simplify its usage we provide:
```@docs
kmpp
kmpp_by_costs
```
