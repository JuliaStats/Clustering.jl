# MCL (Markov Cluster Algorithm)

[Markov Cluster Algorithm](http://micans.org/mcl) works by simulating a
stochastic (Markov) flow in a weighted graph, where each node is a data point,
and the edge weights are defined by the adjacency matrix. ...
When the algorithm converges, it produces the new edge weights that define
the new connected components of the graph (i.e. the clusters).

```@docs
mcl
MCLResult
```
