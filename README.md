# Clustering.jl

This package provides a set of algorithms for data clustering.


# Installation

```julia
Pkg.add("Clustering")
```

# Algorithms

Currently working algorithms:

* Kmeans

To be available:

* K medoids
* Affinity Propagation
* DP means
* ISO Data


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

opts = @options max_iter=50 display=:iter
result = kmeans(x, k, opts)

```

