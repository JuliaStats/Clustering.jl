# [Fuzzy C-means](@id fuzzy_cmeans_def)

[Fuzzy C-means](https://en.wikipedia.org/wiki/Fuzzy_clustering#Fuzzy_C-means_clustering)
is a clustering method that provides cluster membership weights instead
of "hard" classification (e.g. K-means).

From a mathematical standpoint, fuzzy C-means solves the following
optimization
problem:
```math
\arg\min_C \ \sum_{i=1}^n \sum_{j=1}^c w_{ij}^m \| \mathbf{x}_i - \mathbf{c}_{j} \|^2, \
\text{where}\ w_{ij} = \left(\sum_{k=1}^{c} \left(\frac{\left\|\mathbf{x}_i - \mathbf{c}_j \right\|}{\left\|\mathbf{x}_i - \mathbf{c}_k \right\|}\right)^{\frac{2}{m-1}}\right)^{-1}
```

Here, ``\mathbf{c}_j`` is the center of the ``j``-th cluster, ``w_{ij}``
is the membership weight of the ``i``-th point in the ``j``-th cluster,
and ``m > 1`` is a user-defined fuzziness parameter.

```@docs
fuzzy_cmeans
FuzzyCMeansResult
```

## Examples

```@example
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

# get the point memberships over all the clusters
# memberships is a 20x3 matrix
memberships = R.weights
```
