DBSCAN
=========

`Density-based Spatial Clustering of Applications with Noise (DBSCAN) <http://en.wikipedia.org/wiki/DBSCAN>`_ is a data clustering algorithm that finds clusters through density-based expansion of seed points. The algorithm is proposed by:

    Martin Ester, Hans-peter Kriegel, Jörg S, and Xiaowei Xu
    *A density-based algorithm for discovering clusters in large spatial databases with noise.* 
    1996.

**Density Reachability**

DBSCAN's definition of cluster is based on the concept of *density reachability*: a point :math:`q` is said to be *directly density reachable* by another point :math:`p` if the distance between them is below a specified threshold :math:`\epsilon` and :math:`p` is surrounded by sufficiently many points. Then, :math:`q` is considered to be *density reachable* by :math:`p` if there exists a sequence :math:`p_1, p_2, \ldots, p_n` such that :math:`p_1 = p` and :math:`p_{i+1}` is directly density reachable from :math:`p_i`. 

A cluster, which is a subset of the given set of points, satisfies two properties:

1. All points within the cluster are mutually *density-connected*, meaning that for any two distinct points :math:`p` and :math:`q` in a cluster, there exists a point :math:`o` sucht that both :math:`p` and :math:`q` are density reachable from :math:`o`.

2. If a point is density connected to any point of a cluster, it is also part of the cluster. 

**Functions**

The *DBSCAN* algorithm is implemented by the ``dbscan`` function in this package: 

.. function:: dbscan(D, eps, minpts)

    Perform DBSCAN algorithm based on a given distance matrix. 

    :param D: The pairwise distance matrix. ``D[i,j]`` is the distance between points ``i`` and ``j``.
    :param eps: The radius of a neighborhood.
    :param minpts: The minimum number of neighboring points (including self) to qualify a point as a density point. 

    The algorithm returns an instance of ``DbscanResult``, defined as below:

    .. code-block:: julia

        type DbscanResult <: ClusteringResult
            seeds::Vector{Int}          # starting points of clusters, size (k,)
            assignments::Vector{Int}    # assignments, size (n,)
            counts::Vector{Int}         # number of points in each cluster, size (k,)
        end

