K-medoids
===========

`K-medoids <http://en.wikipedia.org/wiki/K-medoids>`_ is a clustering algorithm that seeks a subset of points out of a given set such that the total costs or distances between each point to the closest point in the chosen subset is minimal. This chosen subset of points are called *medoids*.

This package implements a K-means style algorithm instead of PAM, which is considered to be much more efficient and reliable. Particularly, the algorithm is implemented by the ``kmedoids`` function.

.. function:: kmedoids(C, k; ...)

    Performs K-medoids clustering based on a given cost matrix.

    :param C:   The cost matrix, where ``C[i,j]`` is the cost of assigning sample ``j`` to the medoid ``i``.
    :param k:   The number of clusters.

    This function returns an instance of ``KmedoidsResult``, which is defined as follows:

    .. code-block:: julia

        type KmedoidsResult{T} <: ClusteringResult
            medoids::Vector{Int}        # indices of methods (k)
            assignments::Vector{Int}    # assignments (n)
            acosts::Vector{T}           # costs of the resultant assignments (n)
            counts::Vector{Int}         # number of samples assigned to each cluster (k)
            totalcost::Float64          # total assignment cost (i.e. objective) (k)
            iterations::Int             # number of elapsed iterations 
            converged::Bool             # whether the procedure converged
        end

    One may optionally specify some of the options through keyword arguments to control the algorithm:

    ==============  ===========================================================  ========================
     name            description                                                  default
    ==============  ===========================================================  ========================
     ``init``        Initialization algorithm or initial medoids, which can be      ``:kmpp``
                     either of the following:

                     - a symbol indicating the name of seeding algorithm, 
                       ``:rand``, ``:kmpp``, or ``:kmcen`` (see :ref:`cinit`)
                     - an integer vector of length ``k`` that provides the
                       indexes of initial seeds. 
    --------------  -----------------------------------------------------------  ------------------------
     ``maxiter``     Maximum number of iterations.                                ``100``
    --------------  -----------------------------------------------------------  ------------------------
     ``tol``         Tolerable change of objective at convergence.                ``1.0e-6`` 
    --------------  -----------------------------------------------------------  ------------------------
     ``display``     The level of information to be displayed.                    ``:none``
                     (see :ref:`copts`)
    ==============  ===========================================================  ========================


.. function:: kmedoids!(C, medoids, ...)

    Performs K-medoids clustering based on a given cost matrix.

    This function operates on an given set of medoids and updates it inplace. 

    :param C:  The cost matrix, where ``C[i,j]`` is the cost of assigning sample ``j`` to the medoid ``i``.
    :param medoids:  The vector of medoid indexes. The contents of ``medoids`` serve as the initial guess and 
                     will be overrided by the results.

    This function returns an instance of ``KmedoidsResult``.

    One may optionally specify some of the options through keyword arguments to control the algorithm:

    ==============  ===========================================================  ========================
     name            description                                                  default
    ==============  ===========================================================  ========================
     ``maxiter``     Maximum number of iterations.                                ``100``
    --------------  -----------------------------------------------------------  ------------------------
     ``tol``         Tolerable change of objective at convergence.                ``1.0e-6`` 
    --------------  -----------------------------------------------------------  ------------------------
     ``display``     The level of information to be displayed.                    ``:none``
                     (see :ref:`copts`)
    ==============  ===========================================================  ========================    




