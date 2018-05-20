Hierarchical Clustering
========================

`Hierarchical clustering <https://en.wikipedia.org/wiki/Hierarchical_clustering>`_ algorithms build a dendrogram of nested clusters by repeatedly merging or splitting clusters.

**Functions**

.. function:: hclust(D, method)

	Perform hierarchical clustering on distance matrix `D` with specified method.

	:param D: The pairwise distance matrix. ``D[i,j]`` is the distance between points ``i`` and ``j``.
	:param method: A `Symbol` specifying how distance is measured between clusters (which determines the clusters that are merged on each iteration). Valid methods are:
	- ``:single``: use the minimum distance between any of the members
	- ``:average``: use the mean distance between any of the cluster's members
	- ``:complete``: use the maximum distance between any of the members.

The function returns an object of type `Hclust` with the fields
	 - ``merge`` the clusters merged in order.  Leafs are indicated by negative numbers
	 - ``height`` the distance at which the merges take place
	 - ``order`` a preferred grouping for drawing a dendogram.
	 - ``method`` the name of the clustering method.

Example:

	.. code-block:: julia

		D = rand(1000, 1000)
		D += D'  # symmetric distance matrix (optional)
		result = hclust(D, :single)

.. function:: cutree(result; [k], [h])

	Cuts the dendrogram to produce clusters at a specified level of granularity.

	:param result: Object of type ``Hclust`` holding results of a call to ``hclust()``.
	:param k: Integer specifying the number of desired clusters.
	:param h: Integer specifying the height at which to cut the tree.

If both `k` and `h` are specified, it's guaranteed that the number of clusters is ``≥ k`` and their height ``≤ h``.

The output is a vector specifying the cluster index for each datapoint.
