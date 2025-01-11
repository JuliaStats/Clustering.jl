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

# HDBSCAN

Hierarchical Density-based Spatial Clustering of Applications with Noise(HDBSCAN) is similar to DBSCAN but uses hierarchical tree to find clusters. The algorithm was proposed in:

> Ricardo J. G. B. Campello, Davoud Moulavi & Joerg Sander 
> *Density-Based Clustering Based on Hierarchical Density Estimates* 2013

## [Algorithm](@id hdbscan_algorithm)
The main procedure of HDBSCAN:
1. calculate the *mutual reachability distance*
2. generate a *minimum spanning tree*
3. build hierarchy
4. extract the target cluster

### Calculate the *mutual reachability distance*
DBSCAN counts the number of points at a certain distance. But, HDBSCAN uses the opposite way, First, calculate the pairwise distances. Next, defined the core distance(``core_{ncore}(p)``) as a distance with the ncore-th closest point. And then, the mutual reachability distance between point `a` and `b` is defined as: ``d_{mreach-ncore}(a, b) = max\{ core_{ncore}(a), core_{ncore}(b), d(a, b) \}`` where ``d(a, b)`` is a euclidean distance between `a` and `b`(or specified metric).

### Generate a *minimum-spanning tree(MST)*
Conceptually what we will do is the following: consider the data as a weighted graph with the data points as vertices and an edge between any two points with weight equal to the mutual reachability distance of those points.

Then, check the list of edges in descending order of distance whether the points which is connected by it is marked or not. If not, add the edge into the MST. After this processing, all data points are connected to MST with minimum weight.

In practice, this is very expensive since there are ``n^{2}`` edges. 

### Build hierarchy
The next step is to build hierarchy based on MST. This step is very simple: repeatedly unite the clusters which has the minimum edge until all cluster are united into one cluster.

### Extract the target cluster
First we need a different measure than distance to consider the persistence of clusters; instead we will use ``\lambda = \frac{1}{\mathrm{distance}}``.
For a given cluster we can then define values ``\lambda_{\mathrm{birth}}`` and ``\lambda_{\mathrm{death}}`` to be the lambda value when the cluster split off and became it’s own cluster, and the lambda value (if any) when the cluster split into smaller clusters respectively.
In turn, for a given cluster, for each point ``p`` in that cluster we can define the value ``\lambda_{p}`` as the lambda value at which that point ‘fell out of the cluster’ which is a value somewhere between ``\lambda_{\mathrm{birth}}`` and ``\lambda_{\mathrm{death}}`` since the point either falls out of the cluster at some point in the cluster’s lifetime, or leaves the cluster when the cluster splits into two smaller clusters. 
Now, for each cluster compute the stability as

``\sum_{p \in \mathrm{cluster}} (\lambda_{p} - \lambda_{\mathrm{birth}})``.

Declare all leaf nodes to be selected clusters. Now work up through the tree (the reverse topological sort order). If the sum of the stabilities of the child clusters is greater than the stability of the cluster, then we set the cluster stability to be the sum of the child stabilities. If, on the other hand, the cluster’s stability is greater than the sum of its children then we declare the cluster to be a selected cluster and unselect all its descendants. Once we reach the root node we call the current set of selected clusters our flat clustering and return it as the result of clustering.

## Interface
The implementation of *HDBSCAN* algorithm is provided by [`hdbscan`](@ref) function 
```@docs
hdbscan
HdbscanResult
HdbscanCluster
isnoise
```
