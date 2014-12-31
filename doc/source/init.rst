.. _cinit:

Clustering Initialization
==========================

A clustering algorithm usually relies on an initialization scheme to bootstrap the clustering procedure. 

Seeding
--------

*Seeding* is an important family of methods for clustering initialization, which generally refers to an procedure to select a few *seeds* from a data set, each serving as the initial center of a cluster. 

Seeding functions
~~~~~~~~~~~~~~~~~~~

The packages provide two functions ``initseeds`` and ``initseeds_by_costs`` for seeding: 

.. function:: initseeds(algname, X, k)

    Select ``k`` seeds from a given sample matrix ``X``.

    It returns an integer vector of length ``k`` that contains the indexes of chosen seeds. 

    Here, ``algname`` indicates the seeding algorithm, which should be a symbol that may take either of the following values:

    ==============  ============================================================
     algname         description
    ==============  ============================================================
     ``:rand``       Randomly select a subset as seeds
    --------------  ------------------------------------------------------------
     ``:kmpp``       Kmeans++ algorithm, *i.e.* choose seeds sequentially, 
                     the probability of an sample to be chosen is proportional
                     to the minimum cost of assigning it to existing seeds.

                     **Reference:**

                     D. Arthur and S. Vassilvitskii (2007). 
                     *K-means++: the Advantages of Careful Seeding.* 
                     18th Annual ACM-SIAM symposium on Discrete algorithms, 
                     2007.
    --------------  ------------------------------------------------------------
     ``:kmcen``      Choose the ``k`` samples with highest centrality as seeds.
    ==============  ============================================================

.. function:: initseeds_by_costs(algname, C, k)

    Select ``k`` seeds based on a cost matrix ``C``. 

    Here, ``C[i,j]`` is the cost of binding samples ``i`` and ``j`` to the same cluster. One may, for example, use the squared Euclidean distance between samples as the costs.

    The argument ``algname`` determines the choice of algorithm (see above).


In practice, we found that Kmeans++ is the most effective method for initial seeding. Thus, we provide specific functions to simplify the use of Kmeans++ seeding:

.. function:: kmpp(X, k)

    Use *Kmeans++* to choose ``k`` seeds from a data set given by a sample matrix ``X``.

.. function:: kmpp_by_costs(C, k)

    Use *Kmeans++* to choose ``k`` seeds based on a cost matrix ``C``.


Internals
~~~~~~~~~~

In this package, each seeding algorithm is represented by a sub-type of ``SeedingAlgorithm``. Particularly, the random selection algorithm, Kmeans++, and centrality-based algorithm are respectively represented by sub-types ``RandSeedAlg``, ``KmppAlg``, and ``KmCentralityAlg``.

For each sub type, the following methods are implemented:

.. function:: initseeds!(iseeds, alg, X)

    Select seeds from a given sample matrix ``X``, and write the results to ``iseeds``.

    :param iseeds: An pre-allocated array to store the indexes of the chosen seeds.


    :param alg: The algorithm instance. 

    :param X: The given data matrix. Each column of ``X`` is a sample. 

    :return: ``iseeds``

.. function:: initseeds_by_costs!(iseeds, alg, C)

    Select seeds based on a given cost matrix ``C``, and write the results to ``iseeds``.

    :param iseeds: An pre-allocated array to store the indexes of the chosen seeds.

    :param alg: The algorithm instance. 

    :param C: The cost matrix. The value of ``C[i,j]`` is the cost of binding samples ``i`` and ``j`` into the same cluster.

    :return: ``iseeds``

**Note:** For both functions above, the length of ``iseeds`` determines the number of seeds to be selected.

To define a new seeding algorithm, one has to first define a sub type of ``SeedingAlgorithm`` and implement the two functions above.

