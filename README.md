# Clustering.jl

This package provides a set of algorithms for data clustering.


# Installation

```julia
Pkg.add("Clustering")
```

# Algorithms

Currently working algorithms:

* K-means
* Affinity Propagation
* K-medoids

To be available:

* DP means
* ISO Data

# Performance evaluation

For partitioning methods, silhouette widths can be calculated.

# Documentation

## K-means

Interfaces:

```julia
# perform K-means (centers are updated inplace)
result = kmeans!(x, centers, opts) 

# perform K-means based on a given set of inital centers
result = kmeans(x, init_centers, opts)  

# perform K-means to get K centers
result = kmeans(x, k, opts) 
result = kmeans(x, k)

```

All these methods return an instance of ``KmeansResult``, it is defined as

```julia
type KmeansResult{T<:FloatingPoint}
    centers::Matrix{T}         # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{T}           # costs of the resultant assignments (n)
    counts::Vector{Int}        # number of samples assigned to each cluster (k)
    cweights::Vector{T}        # cluster weights (k)
    total_cost::Float64        # total cost (i.e. objective) (k)
    iterations::Int            # number of elapsed iterations 
    converged::Bool            # whether the procedure converged
end
```

Options:

**Note:** options are specified using keyword arguments.

|  name       |  descrption                           | default value |
|-------------|---------------------------------------|---------------|
| max_iters   |  maximum number of iterations         |  100          |
| tol         |  tolerable objv change at convergence |  1.0e-6       |
| weights     |  sample weights (a vector or nothing) |  nothing      |
| display     |  verbosity (``:none``, ``:final``, or ``:iter``)  | ``:iter`` |


### Example

```julia

x = rand(100, 10000)   # a set of 10000 samples (each of dimension 100)

k = 50  # the number of clustering
result = kmeans(x, k; max_iter=50, display=:iter)

```

## Affinity Propagation

Affinity Propagation is an algorithm that uses loopy belief propagation
to run MAP inference to identify some *exemplars*. Unlike kmeans, the
exemplars are chosen from the original samples. After the algorithm
returns, every sample will be assigned to one of the exemplars.

The input of the algorithm is a similarity matrix ``S``.
Unlike kmeans, you don't need to (and cannot) specify the number of
clusters. But the diagonal values of ``S`` will affect how many
clusters you will get at the end. Specifically, ``S[i,j]`` could be
interpreted as the tendency of assigning point ``i`` to point ``j``
(when ``j`` is an exemplar). So

* ``S`` need **NOT** to be symmetric
* ``S[i,i]`` represents the willingness of assigned point ``i`` to
  itself. So generally larger diagonal values for ``S`` means more
  clusters. For example, if ``S[i,i]==max(S)`` for all ``i``, then
  every point will be an exemplar itself.

Usually, assigning the diagonal of ``S`` to be the *median of all the
rest entries* could lead to reasonable results.

Interfaces:

```julia
result = affinity_propagation(S, opts)
```

where the following options could be specified using keyword arguments

```julia
max_iter::Integer = 500,    # max number of iterations
n_stop_check::Integer = 10, # stop if exemplars not changed for this number of iterations
damp::FloatingPoint = 0.5,  # damping factor for message updating, 0 means no damping
display::Symbol = :iter     # whether progress is shown
```

the returning value is a struct that looks like this:

```julia
type AffinityPropagationResult
    exemplar_index ::Vector{Int} # index for exemplars (centers)
    assignments    ::Vector{Int} # assignments for each point
    iterations     ::Int         # number of iterations executed
    converged      ::Bool        # converged or not
end
```

## K medoids

K-medoids is a partitioning algorithm like k-means, but cluster centers are observations
in the data set(,the medoids,) instead of cluster means, and any distance metric can
be used, not necessarily the Euclidean distance.

The input of the algorithm is a dissimilarity matrix `dist`, which should be symmetric, and
the number of clusters, `k`.

Interfaces:

```julia
result = kmedoids(dist, k)
```

The method returns a `KmedoidsResult`:

```julia
type KmedoidsResult
    medoids::Vector{Int}     #indexes of medoids
    assignments::Vector{Int} #cluster assignments
end
```

## Silhouette widths

[Silhouettes](http://en.wikipedia.org/wiki/Silhouette_%28clustering%29) are useful for evaluating
the performance of a partitioning clustering algorithm or to help choose the number of clusters.
The `silhouettes` function takes cluster assignments and a dissimilarity matrix and returns
a vector containing silhouettes for each observation.
