Variation of Information
==========================

`Variation of information <http://en.wikipedia.org/wiki/Variation_of_information>`_ (also known as *shared information distance*) is a measure of the distance between two clusterings. It is devised based on mutual information, but it is a true metric, *i.e.* it satisfies symmetry and triangle inequality. There are variants of variation of information that have been discussed. Vinh, Epps, and Bailey provides a survey of these variants. The names of the variants are taken from Table 3 of this paper.

**References:**

    Meila, Marina (2003). 
    *Comparing Clusterings by the Variation of Information.* 
    Learning Theory and Kernel Machines: 173–187. 

    Vinh, Nguyen Xuan and Epps Julien and Bailey, James (2010)
    *Information Theoretic Measures for Clusterings Comparison:
    Variants, Properties, Normalization and Correction for Chance*
    Journal of Machine Learning Research 11 2837-2854

This package provides the ``varinfo`` function that implements this metric:

.. function:: varinfo(k1, a1, k2, a2, variant)

    Compute the variation of information between two assignments. 

    :param k1: The number of clusters in the first clustering.
    :param a1: The assignment vector for the first clustering.
    :param k2: The number of clusters in the second clustering.
    :param a2: The assignment vector for the second clustering.
    :param variant: The type of normalization to perform defaults to :Djoint which is unnormalized.

    :return: the value of variation of information.

.. function:: varinfo(R, k0, a0, variant)

    This method takes ``R``, an instance of ``ClusteringResult``, as input, and computes the variation of information between its corresponding clustering with one given by ``(k0, a0)``, where ``k0`` is the number of clusters in the other clustering, while ``a0`` is the corresponding assignment vector. 

.. function:: varinfo(R1, R2, variant)

    This method takes ``R1`` and ``R2`` (both are instances of ``ClusteringResult``) and computes the variation of information between them.