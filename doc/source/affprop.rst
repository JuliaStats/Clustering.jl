Affinity Propagation
======================

`Affinity propagation <http://en.wikipedia.org/wiki/Affinity_propagation>`_ is a clustering algorithm based on *message passing* between data points. Similar to *K-medoids*, it finds a subset of points as *exemplars* based on (dis)similarities, and assigns each point in the given data set to the closest exemplar.  

This package implements the affinity propagation algorithm based on the following paper:

    Brendan J. Frey and Delbert Dueck.
    *Clustering by Passing Messages Between Data Points.*
    Science, vol 315, pages 972-976, 2007.

The implementation is optimized by reducing unnecessary array allocation and fusing loops. Specifically, the algorithm is implemented by the ``affinityprop`` function:


.. function:: affinityprop(S; ...)

    Performs affinity propagation based on a similarity matrix ``S``.

    :param S: The similarity matrix. Here, ``S[i,j]`` is the similarity (or negated distance) between samples ``i`` and ``j`` when ``i != j``; while ``S[i,i]`` reflects the *availability* of the ``i``-th sample as an exemplar. 

    This function returns an instance of ``AffinityPropResult``, defined as below:

    .. code-block:: julia

        type AffinityPropResult <: ClusteringResult
            exemplars::Vector{Int}      # indexes of exemplars (centers)
            assignments::Vector{Int}    # assignments for each point
            iterations::Int             # number of iterations executed
            converged::Bool             # converged or not
        end

    One may optionally specify the following keyword arguments:

    ==============  ===========================================================  ========================
     name            description                                                  default
    ==============  ===========================================================  ========================
     ``maxiter``     Maximum number of iterations.                                ``100``
    --------------  -----------------------------------------------------------  ------------------------
     ``tol``         Tolerable change of objective at convergence.                ``1.0e-6`` 
    --------------  -----------------------------------------------------------  ------------------------
     ``damp``        Dampening coefficient.                                       ``0.5``

                     The value should be in ``[0.0, 1.0)``. Larger value of
                     ``damp`` indicates slower (and probably more stable) 
                     update. When ``damp = 0``, it means no dampening 
                     is performed.  
    --------------  -----------------------------------------------------------  ------------------------
     ``display``     The level of information to be displayed                     ``:none``
                     (see :ref:`copts`)                   
    ==============  ===========================================================  ========================
