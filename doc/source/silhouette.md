# Silhouettes

[Silhouettes](http://en.wikipedia.org/wiki/Silhouette_(clustering)) is
a method for evaluating the quality of clustering. Particularly, it provides a
quantitative way to measure how well each point lies within its cluster in
comparison to the other clusters. It was introduced in

> Peter J. Rousseeuw (1987). *Silhouettes: a Graphical Aid to the
> Interpretation and Validation of Cluster Analysis*. Computational and
> Applied Mathematics. 20: 53–65.

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
