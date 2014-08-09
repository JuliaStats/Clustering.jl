Silhouettes
=============

*Silhouettes* is a method for validating clusters of data. Particularly, it provides a quantitative way to measure how well each item lies within its cluster as opposed to others. The *Silhouette* value of a data point is defined as:

.. math::

	s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}

Here, :math:`a(i)` is the average distance from the ``i``-th point to other points within the same cluster. Let :math:`b(i, k)` be the average distance from the ``i``-th point to the points in the ``k``-th cluster. Then :math:`b(i)` is the minimum of all :math:`b(i, k)` over all clusters that the ``i``-th point is not assigned to.

Note that the value of :math:`s(i)` is not greater than one, and that :math:`s(i)` is close to one indicates that the ``i``-th point lies well within its own cluster.

.. function:: silhouettes(assignments, counts, dists)

	Compute silhouette values for individual points w.r.t. a given clustering.

	:param assignments: the vector of assignments
	:param counts: the number of points falling in each cluster
	:param dists: the pairwise distance matrix

	:return: It returns a vector of silhouette values for individual points. In practice, one may use the average of these silhouette values to assess given clustering results.

.. function:: silhouettes(R, dists)

	This method accepts a clustering result ``R`` (of a sub-type of ``ClusteringResult``).

	It is equivalent to ``silhouettes(assignments(R), counts(R), dists)``.

