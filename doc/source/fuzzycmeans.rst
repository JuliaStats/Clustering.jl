Fuzzy C-means
=============

`Fuzzy C-means <https://en.wikipedia.org/wiki/Fuzzy_clustering#Fuzzy_C-means_clustering>`_ is a clustering method that provides cluster membership weights instead of "hard" classification (e.g. K-means).

From a mathematical standpoint, Fuzzy C-means solves the following optimization problem:

.. math::

    \arg\min_C \ \sum_{i=1}^n \sum_{j=1}^c w_{ij}^m \| \mathbf{x}_i - \mathbf{c}_{j} \|^2

    \mathrm{s.t.} \ w_{ij} = \frac{1}{\sum_{k=1}^{c} \left(\frac{\left\|\mathbf{x}_i - \mathbf{c}_j \right\|}{\left\|\mathbf{x}_i - \mathbf{c}_k \right\|}\right)^{\frac{2}{m-1}}}

Here, :math:`\mathbf{c}_j` is the center of the :math:`j`-th cluster, :math:`w_{ij}` is the membership weight of point :math:`i` in cluster :math:`j`, and :math:`m` is a user-defined fuzziness parameter. 



.. function:: fuzzy_cmeans(data, C, fuzziness; ...)

    Performs Fuzzy C-means clustering over the given dataset.

    :param data:           The given :math:`d \times n` sample matrix. Each column of ``data`` is a sample. 
    :param C:              The number of clusters, :math:`2 \le C < n`.
    :param fuzziness:      Clusters fuzziness (see the `m` in the mathematical formulation), ``fuzziness`` > 1.

    This function returns an instance of ``FuzzyCMeansResult``:

    .. code-block:: julia

        struct FuzzyCMeansResult{T<:AbstractFloat} <: ClusteringResult
            centers::Matrix{T}          # cluster centers (d x C)
            weights::Matrix{Float64}    # assigned weights (n x C)
            iterations::Int             # number of elasped iterations
            converged::Bool             # whether the procedure converged
        end

    One may optionally specify some of the options through keyword arguments to control the algorithm:

    ================  ===========================================================  ========================
     name              description                                                  default
    ================  ===========================================================  ========================
     ``maxiter``       Maximum number of iterations.                                ``100``
    ----------------  -----------------------------------------------------------  ------------------------
     ``tol``           Absolute convergence tolerance of clusters centers.          ``1.0e-3`` 
    ----------------  -----------------------------------------------------------  ------------------------
     ``dist_metric``   Distance used to get a notion of closeness between points    ``Euclidean``
    ----------------  -----------------------------------------------------------  ------------------------
     ``display``       The level of information to be displayed.                    ``:none``
                       (see :ref:`copts`)
    ================  ===========================================================  ========================

**Examples:**

.. code-block:: julia

    using Clustering

    # make a random dataset with 1000 points
    # each point is a 5-dimensional vector
    X = rand(5, 1000)

    # performs Fuzzy C-means over X, trying to group them into 3 clusters
    # with a fuzziness factor of 2. Set maximum number of iterations to 200
    # set display to :iter, so it shows progressive info at each iteration
    R = fuzzy_cmeans(X, 3, 2, maxiter=200, display=:iter)

    # get the centers (i.e. weighted mean vectors)
    # M is a 5x3 matrix
    # M[:, k] is the center of the k-th cluster
    M = R.centers

    # get the samples memberships over all the clusters
    # memberships is a 20x3 matrix
    memberships = R.weights

