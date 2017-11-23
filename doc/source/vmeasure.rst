V-measure
=============

The V-Measure is defined as the harmonic mean of homogeneity :math:`h` and completeness :math:`c` of the clustering. Both these measures can be expressed in terms of the mutual information and entropy measures originating from the field of information retrieval.

.. math::

	V_\beta = (1+\beta)\frac{h \cdot c}{\beta \cdot h + c}

Homogeneity :math:`h` is maximized when each cluster contains elements of as few different classes as possible. Completeness :math:`c` aims to put all elements of each class in single clusters.

**References:**

    Andrew Rosenberg and Julia Hirschberg, 2007. "V-Measure: A conditional entropy-based external cluster evaluation measure"

This package provides the ``vmeasure`` function that implements this metric:

.. function:: vmeasure(assign1, assign2; β =1.0 )

	Compute V-measure value between two clustering assignments.

	:param assign1: the vector of assignments for the first clustering.
	:param assign2: the vector of assignments for the second clustering.
	:param β: the weight of harmonic mean of homogeneity and completeness.

	:return: It returns a v-measure value.

.. function:: vmeasure(R, assign)

    This method takes ``R``, an instance of ``ClusteringResult``, and the corresponding assignment vector ``assign`` as input, and computes v-measure value (see above).

.. function:: vmeasure(R1, R2)

    This method takes ``R1`` and ``R2`` (both are instances of ``ClusteringResult``) and computes v-measure value (see above).

	It is equivalent to ``vmeasure(assignments(R1), assignments(R1))``.
