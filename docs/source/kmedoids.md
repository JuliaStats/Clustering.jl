# K-medoids

[K-medoids](http://en.wikipedia.org/wiki/K-medoids) is a clustering
algorithm that works by finding ``k`` data points (called *medoids*)
such that the total distance between each data point and the closest
*medoid* is minimal.

```@docs
kmedoids
kmedoids!
KmedoidsResult
```

## [References](@id kmedoid_refs)
1. Teitz, M.B. and Bart, P. (1968). *Heuristic Methods for Estimating the Generalized Vertex Median of a Weighted Graph*. Operations Research, 16(5), 955–961.
   [doi:10.1287/opre.16.5.955](https://doi.org/10.1287/opre.16.5.955)
2. Schubert, E. and Rousseeuw, P.J. (2019). *Faster k-medoids clustering: Improving the PAM, CLARA, and CLARANS Algorithms*. SISAP, 171-187.
   [doi:10.1007/978-3-030-32047-8_16](https://doi.org/10.1007/978-3-030-32047-8_16)
