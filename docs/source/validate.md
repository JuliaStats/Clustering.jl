# [Evaluation & Validation](@id clu_validate)

*Clustering.jl* package provides a number of methods to compare different clusterings,
evaluate clustering quality or validate its correctness.

## Clustering comparison

Methods to compare two clusterings and measure their similarity.

### Cross tabulation

[Cross tabulation](https://en.wikipedia.org/wiki/Contingency_table), or
*contingency matrix*, is a basis for many clustering quality measures.
It shows how similar are the two clusterings on a cluster level.

*Clustering.jl* extends `StatsBase.counts()` with methods that accept
[`ClusteringResult`](@ref) arguments:
```@docs
counts(::ClusteringResult, ::ClusteringResult)
```

### Confusion matrix

[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
for the two clusterings is a 2×2 contingency table that counts
how frequently the pair of data points are in the same or different clusters.

```@docs
confusion
```

### Rand index

[Rand index](http://en.wikipedia.org/wiki/Rand_index) is a measure of
the similarity between the two data clusterings. From a mathematical
standpoint, Rand index is related to the prediction accuracy, but is applicable
even when the original class labels are not used.

```@docs
randindex
```

### Variation of Information

[Variation of information](http://en.wikipedia.org/wiki/Variation_of_information)
(also known as *shared information distance*) is a measure of the
distance between the two clusterings. It is devised from the *mutual
information*, but it is a true metric, *i.e.* it is symmetric and satisfies
the triangle inequality.

```@docs
Clustering.varinfo
```
### V-measure

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

### Mutual information

[Mutual information](https://en.wikipedia.org/wiki/Mutual_information)
quantifies the "amount of information" obtained about one random variable
through observing the other random variable. It is used in determining
the similarity of two different clusterings of a dataset.

```@docs
mutualinfo
```

## Clustering quality indices

[`clustering_quality()`](@ref clustering_quality) methods allow computing *intrinsic* clustering quality indices,
i.e. the metrics that depend only on the clustering itself and do not use the external knowledge.
These metrics can be used to compare different clustering algorithms or choose the optimal number of clusters.

|   **quality index**                         |   **`quality_index` option**  |  **clustering type**  | **better quality** | **cluster centers** |
|:-------------------------------------------:|:--------------------:|:----------:|:-------------:|:-------------------:|
| [Calinski-Harabasz](@ref calinsky_harabasz) | `:calinsky_harabasz` | hard/fuzzy | *higher* values |       required      |
| [Xie-Beni](@ref xie_beni)                   | `:xie_beni`          | hard/fuzzy | *lower* values  |       required      |
| [Davis-Bouldin](@ref davis_bouldin)         | `:davis_bouldin`     |    hard    | *lower* values  |       required      |
| [Dunn](@ref dunn)                           | `:dunn`              |    hard    | *higher* values |     not required    |
| [silhouettes](@ref silhouettes_index)       | `:silhouettes`       |    hard    | *higher* values |     not required    |

```@docs
clustering_quality
```

The clustering quality index definitions use the following notation:
- ``x_1, x_2, \ldots, x_n``: data points,
- ``C_1, C_2, \ldots, C_k``: clusters,
- ``c_j`` and ``c``: cluster centers and global dataset center,
- ``d``: a similarity (distance) function,
- ``w_{ij}``: weights measuring membership of a point ``x_i`` to a cluster ``C_j``,
- ``\alpha``:  a fuzziness parameter.

### [Calinski-Harabasz index](@id calinsky_harabasz)

[*Calinski-Harabasz* index](https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index) (option `:calinski_harabasz`)
measures corrected ratio between global inertia of the cluster centers and the summed internal inertias of clusters:
```math
\frac{n-k}{k-1}\frac{\sum_{C_j}|C_j|d(c_j,c)}{\sum\limits_{C_j}\sum\limits_{x_i\in C_j} d(x_i,c_j)} \quad \text{and}\quad
\frac{n-k}{k-1} \frac{\sum\limits_{C_j}\left(\sum\limits_{x_i}w_{ij}^\alpha\right) d(c_j,c)}{\sum_{C_j} \sum_{x_i} w_{ij}^\alpha d(x_i,c_j)}
```
for hard and fuzzy (soft) clusterings, respectively.
*Higher* values indicate better quality.

### [Xie-Beni index](@id xie_beni)

*Xie-Beni* index (option `:xie_beni`) measures ratio between summed inertia of clusters
and the minimum distance between cluster centres:
```math
\frac{\sum_{C_j}\sum_{x_i\in C_j}d(x_i,c_j)}{n\min\limits_{c_{j_1}\neq c_{j_2}} d(c_{j_1},c_{j_2}) }
\quad \text{and}\quad
\frac{\sum_{C_j}\sum_{x_i} w_{ij}^\alpha d(x_i,c_j)}{n\min\limits_{c_{j_1}\neq c_{j_2}} d(c_{j_1},c_{j_2}) }
```
for hard and fuzzy (soft) clusterings, respectively.
*Lower* values indicate better quality.

### [Davis-Bouldin index](@id davis_bouldin)
[*Davis-Bouldin* index](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index)
(option `:davis_bouldin`) measures average cohesion based on the cluster diameters and distances between cluster centers:
```math
\frac{1}{k}\sum_{C_{j_1}}\max_{c_{j_2}\neq c_{j_1}}\frac{S(C_{j_1})+S(C_{j_2})}{d(c_{j_1},c_{j_2})}
```
where
```math
S(C_j) = \frac{1}{|C_j|}\sum_{x_i\in C_j}d(x_i,c_j).
```
*Lower* values indicate better quality.

### [Dunn index](@id dunn)
[*Dunn* index](https://en.wikipedia.org/wiki/Dunn_index) (option `:dunn`)
measures the ratio between the nearest neighbour distance divided by the maximum cluster diameter:
```math
\frac{\min\limits_{ C_{j_1}\neq C_{j_2}} \mathrm{dist}(C_{j_1},C_{j_2})}{\max\limits_{C_j}\mathrm{diam}(C_j)}
```
where
```math
\mathrm{dist}(C_{j_1},C_{j_2}) = \min\limits_{x_{i_1}\in C_{j_1},x_{i_2}\in C_{j_2}} d(x_{i_1},x_{i_2}),\quad \mathrm{diam}(C_j) = \max\limits_{x_{i_1},x_{i_2}\in C_j} d(x_{i_1},x_{i_2}).
```
It is more computationally demanding quality index, which can be used when the centres are not known. *Higher* values indicate better quality.

### [Silhouettes](@id silhouettes_index)

[*Silhouettes* metric](http://en.wikipedia.org/wiki/Silhouette_(clustering)) quantifies the correctness of point-to-cluster asssignment by
comparing the distance of the point to its cluster and to the other clusters.

The *Silhouette* value for the ``i``-th data point is:
```math
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}, \ \text{where}
```
 - ``a_i`` is the average distance from the ``i``-th point to the other points in
   the *same* cluster ``z_i``,
 - ``b_i ≝ \min_{k \ne z_i} b_{ik}``, where ``b_{ik}`` is the average distance
   from the ``i``-th point to the points in the ``k``-th cluster.

Note that ``s_i \le 1``, and that ``s_i`` is close to ``1`` when the ``i``-th
point lies well within its own cluster. This property allows using average silhouette value
`mean(silhouettes(assignments, counts, X))` as a measure of clustering quality;
it is also available using [`clustering_quality(...; quality_index = :silhouettes)`](@ref clustering_quality) method.
Higher values indicate better separation of clusters w.r.t. point distances.

```@docs
silhouettes
```

[`clustering_quality(..., quality_index=:silhouettes)`](@ref clustering_quality)
provides mean silhouette metric for the datapoints. Higher values indicate better quality.

## References
> Olatz Arbelaitz *et al.* (2013). *An extensive comparative study of cluster validity indices*. Pattern Recognition. 46 1: 243-256. [doi:10.1016/j.patcog.2012.07.021](https://doi.org/10.1016/j.patcog.2012.07.021)

> Aybükë Oztürk, Stéphane Lallich, Jérôme Darmont. (2018). *A Visual Quality Index for Fuzzy C-Means*.  14th International Conference on Artificial Intelligence Applications and Innovations (AIAI 2018). 546-555. [doi:10.1007/978-3-319-92007-8_46](https://doi.org/10.1007/978-3-319-92007-8_46).

### Examples

Exemplary data with 3 real clusters.
```@example clu_quality
using Plots, Plots.PlotMeasures, Clustering
X_clusters = [(center = [4., 5.], std = 0.4, n = 10),
              (center = [9., -5.], std = 0.4, n = 5),
              (center = [-4., -9.], std = 1, n = 5)]
X = mapreduce(hcat, X_clusters) do (center, std, n)
    center .+ std .* randn(length(center), n)
end
X_assignments = mapreduce(vcat, enumerate(X_clusters)) do (i, (_, _, n))
    fill(i, n)
end

scatter(view(X, 1, :), view(X, 2, :),
    markercolor = X_assignments,
    plot_title = "Data", label = nothing,
    xlabel = "x", ylabel = "y",
    legend = :outerright,
    size = (600, 500)
);
savefig("clu_quality_data.svg"); nothing # hide
```
![](clu_quality_data.svg)

Hard clustering quality for [K-means](@ref) method with 2 to 5 clusters:

```@example clu_quality
hard_nclusters = 2:5
clusterings = kmeans.(Ref(X), hard_nclusters)

plot((
    plot(hard_nclusters,
         clustering_quality.(Ref(X), clusterings, quality_index = qidx),
         marker = :circle,
         title = ":$qidx", label = nothing,
    ) for qidx in [:silhouettes, :calinski_harabasz, :xie_beni, :davies_bouldin, :dunn])...,
    layout = (2, 3),
    xaxis = "N clusters", yaxis = "Quality",
    plot_title = "\"Hard\" clustering quality indices",
    size = (1000, 600), left_margin = 10pt
)
savefig("clu_quality_hard.svg"); nothing # hide
```
![](clu_quality_hard.svg)

Fuzzy clustering quality for fuzzy C-means method with 2 to 5 clusters:
```@example clu_quality
fuzziness = [1.3 2 3]
fuzzy_nclusters = 2:5
fuzzy_clusterings = fuzzy_cmeans.(Ref(X), fuzzy_nclusters, fuzziness)

plot((
    plot(fuzzy_nclusters,
         [clustering_quality.(Ref(X), fuzz_clusterings,
                              fuzziness = fuzz, quality_index = qidx)
          for (fuzz, fuzz_clusterings) in zip(fuzziness, eachcol(fuzzy_clusterings))],
         marker = :circle,
         title = ":$qidx", label = ["Fuzziness $fuzz" for fuzz in fuzziness],
    ) for qidx in [:calinski_harabasz, :xie_beni])...,
    layout = (1, 2), legend = :left,
    xaxis = "N clusters", yaxis = "Quality",
    plot_title = "\"Soft\" clustering quality indices",
    size = (700, 350), left_margin = 10pt
)
savefig("clu_quality_soft.svg"); nothing # hide
```
![](clu_quality_soft.svg)


## Other packages

* [ClusteringBenchmarks.jl](https://github.com/HolyLab/ClusteringBenchmarks.jl) provides
  benchmark datasets and implements additional methods for evaluating clustering performance.
