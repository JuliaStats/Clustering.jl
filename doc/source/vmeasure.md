# V-measure

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
