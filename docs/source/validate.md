# [Evaluation & Validation](@id clu_validate)

*Clustering.jl* package provides a number of methods to evaluate the results of
a clustering algorithm and/or to validate its correctness.


## Cross tabulation

[Cross tabulation](https://en.wikipedia.org/wiki/Contingency_table), or
*contingency matrix*, is a basis for many clustering quality measures.
It shows how similar are the two clusterings on a cluster level.

*Clustering.jl* extends `StatsBase.counts()` with methods that accept
[`ClusteringResult`](@ref) arguments:
```@docs
counts(a::ClusteringResult, b::ClusteringResult)
```


## Rand index

[Rand index](http://en.wikipedia.org/wiki/Rand_index) is a measure of
the similarity between the two data clusterings. From a mathematical
standpoint, Rand index is related to the prediction accuracy, but is applicable
even when the original class labels are not used.

```@docs
randindex
```


## Silhouettes

[Silhouettes](http://en.wikipedia.org/wiki/Silhouette_(clustering)) is
a method for evaluating the quality of clustering. Particularly, it provides a
quantitative way to measure how well each point lies within its cluster in
comparison to the other clusters.

The *Silhouette* value for the ``i``-th data point is:
```math
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}, \ \text{where}
```
 - ``a_i`` is the average distance from the ``i``-th point to the other points in
   the same cluster ``z_i``,
 - ``b_i ≝ \min_{k \ne z_i} b_{ik}``, where ``b_{ik}`` is the average distance
   from the ``i``-th point to the points in the ``k``-th cluster.

Note that ``s_i \le 1``, and that ``s_i`` is close to ``1`` when the ``i``-th
point lies well within its own cluster. This property allows using
`mean(silhouettes(assignments, counts, X))` as a measure of clustering quality.
Higher values indicate better separation of clusters w.r.t. point distances.

```@docs
silhouettes
```

## Clustering quality indices

A group of clustering evaluation metrics which are intrinsic, i.e. depend only on the clustering itself. They can be used to compare different clustering algorithms or choose the optimal number of clusters.

The data points are denoted by ``x_1,x_2,\ldots, x_n``, clusters by ``C_1,C_2,\ldots,C_k`` and their centers by ``c_j``, ``c`` is global center of the dataset, ``d`` is a given similarity (distance) function. For soft (fuzzy) clustering ``w_{ij}`` are weights measuring membership of point ``x_i`` to cluster ``C_j`` and ``m`` is the fuzziness parameter.  Arrow up (↑) or down (↓) indicate if higher or lower index values indicate better quality.

Given this notation, available indices and their definitions are:

### Average silhouette index (↑) 

Option `:silhouettes`. The average over all silhouettes in the data set, see section **Silhouettes** for a more detailed description of the method.

### Calinski-Harabasz index (↑) 

Option `:calinski_harabasz`. Measures corrected ratio between the summed internal inertia of clusters divided by global inertia of the cluster centers. For hard clustering and soft (fuzzy) it is defined as

```math

\frac{n-k}{k-1}\frac{\sum_{C_j}|C_j|d(c_j,c)}{\sum\limits_{C_j}\sum\limits_{x_i\in C_j} d(x_i,c_j)} \quad \text{and}\quad 
\frac{n-k}{k-1} \frac{\sum_{C_j}\sum_{x_i} w_{ik}^md(x_i,c_j)}{\sum\limits_{C_j}\sum\limits_{x_i}w_{ij}^m d(c_j,c)}
```
respectively.


### Xie-Beni index (↓)
Option `:xie_beni`. Measures ratio between summed inertia of clusters and minimum distance between cluster centres. For hard clustering and soft (fuzzy) clustering. It is defined as
```math
\frac{\sum_{C_j}\sum_{x_i\in C_j}d(x_i,c_j)}{n\min\limits_{c_{j_1}\neq c_{j_2}} d(c_{j_1},c_{j_2}) }
\quad \text{and}\quad
\frac{\sum_{C_j}\sum_{x_i} w_{ij}^md(x_i,c_j)}{n\min\limits_{c_{j_1}\neq c_{j_2}} d(c_{j_1},c_{j_2}) }
```
respectively.
### [Davis-Bouldin index](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index) (↓)
Option `:davis_bouldin`. It measures average cohesion based on the cluster diameters and distances between cluster centers. It is defined as

```math
\frac{1}{k}\sum_{C_{j_1}}\max_{c_{j_2}\neq c_{j_1}}\frac{S(C_{j_1})+S(C_{j_2})}{d(c_{j_1},c_{j_2})}
```
where
```math
S(C_j) = \frac{1}{|C_j|}\sum_{x_i\in C_j}d(x_i,c_j).
```
### [Dunn index](https://en.wikipedia.org/wiki/Dunn_index) (↑) 
Option `:dunn`. More computationally demanding index which can be used when the centres are not known. It measures ratio between the nearest neighbour distance divided by the maximum cluster diameter. It is defined as
```math
\frac{\min\limits_{ C_{j_1}\neq C_{j_2}} \delta(C_{j_1},C_{j_2})}{\max\limits_{C_j}\Delta(C_j)}
```

where
```math
\delta(C_{j_1},C_{j_2}) = \min\limits_{x_{i_1}\in C_{j_1},x_{i_2}\in C_{j_2}} d(x_{i_1},x_{i_2}),\quad \Delta(C_j) = \max\limits_{x_{i_1},x_{i_2}\in C_j} d(x_{i_1},x_{i_2}).
```

### References
> Olatz Arbelaitz *et al.* (2013). *An extensive comparative study of cluster validity indices*. Pattern Recognition. 46 1: 243-256. [doi:10.1016/j.patcog.2012.07.021](https://doi.org/10.1016/j.patcog.2012.07.021)

> Aybükë Oztürk, Stéphane Lallich, Jérôme Darmont. (2018). *A Visual Quality Index for Fuzzy C-Means*.  14th International Conference on Artificial Intelligence Applications and Innovations (AIAI 2018). 546-555. [doi:10.1007/978-3-319-92007-8_46](https://doi.org/10.1007/978-3-319-92007-8_46). 

### Examples

Exemplary data with 3 clusters. 
```@example
using Plots, Clustering
X = hcat([4., 5.] .+ 0.4 * randn(2, 10),
         [9., -5.] .+ 0.4 * randn(2, 5),
         [-4., -9.] .+ 1 * randn(2, 5))


scatter(X[1,:],X[2,:],
    label = "exemplary data points",
    xlabel = "x",
    ylabel = "y",
    legend = :right,
)
```

Hard clustering quality for number of clusters in `2:5`

```@example 
using Plots, Clustering
X = hcat([4., 5.] .+ 0.4 * randn(2, 10),
         [9., -5.] .+ 0.4 * randn(2, 5),
         [-4., -9.] .+ 1 * randn(2, 5))

clusterings = kmeans.(Ref(X), 2:5)
hard_indices = [:silhouette, :calinski_harabasz, :xie_beni, :davies_bouldin, :dunn]

kmeans_quality = 
    Dict(qidx => clustering_quality.(Ref(X), clusterings, quality_index = qidx)
        for qidx in hard_indices
    )

p = [
    plot(2:5, kmeans_quality[qidx],
        marker = :circle,
        title = string.(qidx),
        label = nothing,
    )
        for qidx in hard_indices
]
plot(p...,
    layout = (3,2),
    plot_title = "Quality indices for various number of clusters"
)
```

Soft clustering quality for number of clusters in `2:5`
```@example
using Plots, Clustering
X = hcat([4., 5.] .+ 0.4 * randn(2, 10),
         [9., -5.] .+ 0.4 * randn(2, 5),
         [-4., -9.] .+ 1 * randn(2, 5))

fuzziness = 2
soft_indices = [:calinski_harabasz, :xie_beni]
fuzzy_clusterings = fuzzy_cmeans.(Ref(X), 2:5, fuzziness)

fuzzy_cmeans_quality = 
    Dict(qidx => clustering_quality.(Ref(X), fuzzy_clusterings, fuzziness, quality_index = qidx)
        for qidx in soft_indices
    )


p = [
    plot(2:5, fuzzy_cmeans_quality[qidx],
        marker = :circle,
        title = string.(qidx),
        label = nothing,
    )
        for qidx in soft_indices
]
plot(p...,
    layout = (2,1),
    plot_title = "Quality indices for various number of clusters"
)

```

```@docs
clustering_quality
```

## Variation of Information

[Variation of information](http://en.wikipedia.org/wiki/Variation_of_information)
(also known as *shared information distance*) is a measure of the
distance between the two clusterings. It is devised from the *mutual
information*, but it is a true metric, *i.e.* it is symmetric and satisfies
the triangle inequality.

```@docs
varinfo
```


## V-measure

*V*-measure can be used to compare the clustering results with the
existing class labels of data points or with the alternative clustering.
It is defined as the harmonic mean of homogeneity (``h``) and completeness
(``c``) of the clustering:
```math
V_{\beta} = (1+\beta)\frac{h \cdot c}{\beta \cdot h + c}.
```
Both ``h`` and ``c`` can be expressed in terms of the mutual information and
entropy measures from the information theory. Homogeneity (``h``) is maximized
when each cluster contains elements of as few different classes as possible.
Completeness (``c``) aims to put all elements of each class in single clusters.
The ``\beta`` parameter (``\beta > 0``) could used to control the weights of
``h`` and ``c`` in the final measure. If ``\beta > 1``, *completeness* has more
weight, and when ``\beta < 1`` it's *homogeneity*.

```@docs
vmeasure
```

## Mutual information

[Mutual information](https://en.wikipedia.org/wiki/Mutual_information)
quantifies the "amount of information" obtained about one random variable
through observing the other random variable. It is used in determining
the similarity of two different clusterings of a dataset.

```@docs
mutualinfo
```

## Confusion matrix

Pair [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
arising from two clusterings is a 2×2 contingency table representation of
the partition co-occurrence, see [`counts`](@ref).

```@docs
confusion
```
