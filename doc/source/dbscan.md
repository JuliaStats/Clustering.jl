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

A cluster, which is a subset of the given set of points, satisfies two
properties:
 1. All points within the cluster are mutually *density-connected*,
    meaning that for any two distinct points ``p`` and ``q`` in a
    cluster, there exists a point ``o`` sucht that both ``p`` and ``q``
    are density reachable from ``o``.
 2. If a point is density-connected to any point of a cluster, it is
    also part of that cluster.

## Interface

There are two implementations of *DBSCAN* algorithm in this package
(both provided by [`dbscan`](@ref) function):
 - Distance (adjacency) matrix-based. It requires ``O(N^2)`` memory to run.
   Boundary points cannot be shared between the clusters.
 - Adjacency list-based. The input is the ``d \times n`` matrix of point
   coordinates. The adjacency list is built on the fly. The performance is much
   better both in terms of running time and memory usage. Returns a vector of
   [`DbscanCluster`](@ref) objects that contain the indices of the *core* and
   *boundary* points, making it possible to share the boundary points between
   multiple clusters.

```@docs
dbscan
DbscanResult
DbscanCluster
```
