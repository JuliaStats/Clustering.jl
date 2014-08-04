Overview
==========

*Clustering.jl* provides functionalities in three aspects that are related to data clustering:

- Clustering initialization, *e.g.* K-means++ seeding.
- Clustering algorithms, *e.g.* K-means, K-medoids, Affinity propagation, and DBSCAN, etc. 
- Clustering evaluation, *e.g.* Silhouettes and variational information.

**Inputs**

A clustering algorithm, depending on its nature, may accept an input matrix in either of the following forms:

- Sample matrix ``X``, where each column ``X[:,i]`` corresponds to an observed sample.
- Distance matrix ``D``, where ``D[i,j]`` indicates the distance between samples ``i`` and ``j``, or the cost of assigning one to the other.


**Results**

A clustering algorithm would return a struct that captures both the clustering results (*e.g.* assignments of samples to clusters) and information about the clustering procedure (*e.g.* the number of iterations or whether the iterative update converged). 

Generally, the resultant struct is defined as an instance of a sub-type of ``ClusteringResult``. The following generic methods are implemented for these subtypes (let ``R`` be an instance):

.. function:: nclusters(R)

	Get the number of clusters

.. function:: assignments(R)

	Get a vector of assignments. 

	Let ``a = assignments(R)``, then ``a[i]`` is the index of the cluster to which the ``i``-th sample is assigned.

.. function:: counts(R)

	Get sample counts of clusters. 

	Let ``c = counts(R)``, then ``c[k]`` is the number of samples assigned to the ``k``-th cluster.

