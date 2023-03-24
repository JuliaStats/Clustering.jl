# Affinity Propagation

[Affinity propagation](http://en.wikipedia.org/wiki/Affinity_propagation) is a
clustering algorithm based on *message passing* between data points.
Similar to [K-medoids](@ref), it looks at the (dis)similarities in the data,
picks one *exemplar* data point for each cluster, and assigns every point in the
data set to the cluster with the closest *exemplar*.

```@docs
affinityprop
AffinityPropResult
```
