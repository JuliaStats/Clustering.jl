Rand indices
==========================

`Rand index <http://en.wikipedia.org/wiki/Rand_index>`_ is a measure of the similarity between two data clusterings. From a mathematical standpoint, Rand index is related to the accuracy, but is applicable even when class labels are not used.

**References:**

    Lawrence Hubert and Phipps Arabie (1985). 
    *Comparing partitions.* 
    Journal of Classification 2 (1): 193–218 

    Meila, Marina (2003). 
    *Comparing Clusterings by the Variation of Information.* 
    Learning Theory and Kernel Machines: 173–187.

This package provides the ``randindex`` function that implements several metrics:

.. function:: randindex(c1, c2)

    Compute the tuple of indices (Adjusted Rand index, Rand index, Mirkin's index, Hubert's index) between two assignments. 

    :param c1: The assignment vector for the first clustering.
    :param c2: The assignment vector for the second clustering.

    :return: tuple of indices.

.. function:: varinfo(R, k0, a0)

    This method takes ``R``, an instance of ``ClusteringResult``, as input, and computes the tuple of indices (see above) where ``a0`` is the corresponding assignment vector. 

.. function:: varinfo(R1, R2)

    This method takes ``R1`` and ``R2`` (both are instances of ``ClusteringResult``) and computes  the tuple of indices (see above) between them.