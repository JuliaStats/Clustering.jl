Variation of Information
==========================

`Variation of information <http://en.wikipedia.org/wiki/Variation_of_information>`_ (also known as *shared information distance*) is a measure of the distance between two clusterings. It is devised based on mutual information, but it is a true metric, *i.e.* it satisfies symmetry and triangle inequality. 

**References:**

    Meila, Marina (2003). 
    *Comparing Clusterings by the Variation of Information.* 
    Learning Theory and Kernel Machines: 173–187. 

This package provides the ``varinfo`` function that implements this metric:

.. function:: varinfo(k1, a1, k2, a2)

    Compute the variation of information between two assignments. 

    :param k1: The number of clusters in the first clustering.
    :param a1: The assignment vector for the first clustering.
    :param k2: The number of clusters in the second clustering.
    :param a2: The assignment vector for the second clustering.

    :return: the value of variation of information.

.. function:: varinfo(R, k0, a0)

    This method takes ``R``, an instance of ``ClusteringResult``, as input, and computes the variation of information between its corresponding clustering with one given by ``(k0, a0)``, where ``k0`` is the number of clusters in the other clustering, while ``a0`` is the corresponding assignment vector. 

.. function:: varinfo(R1, R2)

    This method takes ``R1`` and ``R2`` (both are instances of ``ClusteringResult``) and computes the variation of information between them.

