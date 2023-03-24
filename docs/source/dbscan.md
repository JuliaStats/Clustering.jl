# DBSCAN

[Density-based Spatial Clustering of Applications with Noise
(DBSCAN)](http://en.wikipedia.org/wiki/DBSCAN) is a data clustering
algorithm that finds clusters through density-based expansion of seed
points. The algorithm was proposed in:

> Martin Ester, Hans-peter Kriegel, Jörg S, and Xiaowei Xu *A
> density-based algorithm for discovering clusters in large spatial
> databases with noise.* 1996.

## Density Reachability

DBSCAN's definition of a cluster is based on the concept of *density
reachability*: a point ``q`` is said to be *directly density reachable*
by another point ``p`` if the distance between them is below a specified
threshold ``\epsilon`` and ``p`` is surrounded by sufficiently many
points. Then, ``q`` is considered to be *density reachable* by ``p`` if
there exists a sequence ``p_1, p_2, \ldots, p_n`` such that ``p_1 = p``
and ``p_{i+1}`` is directly density reachable from ``p_i``.

The points within DBSCAN clusters are categorized into *core* (or *seeds*)
and *boundary*:
 1. All points of the cluster *core* are mutually *density-connected*,
    meaning that for any two distinct points ``p`` and ``q`` in a
    core, there exists a point ``o`` such that both ``p`` and ``q``
    are *density reachable* from ``o``.
 2. If a point is *density-connected* to any point of a cluster core, it is
    also part of the core.
 3. All points within the ``\epsilon``-neighborhood of any core point, but
    not belonging to that core (i.e. not *density reachable* from the core),
    are considered cluster *boundary*.

## Interface

The implementation of *DBSCAN* algorithm provided by [`dbscan`](@ref) function
supports the two ways of specifying clustering data:
 - The ``d \times n`` matrix of point coordinates. This is the preferred method
   as it uses memory- and time-efficient neighboring points queries via
   [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) package.
 - The ``n\times n`` matrix of precalculated pairwise point distances.
   It requires ``O(n^2)`` memory and time to run.

```@docs
dbscan
DbscanResult
DbscanCluster
```
