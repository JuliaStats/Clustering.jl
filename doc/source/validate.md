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
