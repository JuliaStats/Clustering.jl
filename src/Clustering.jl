module Clustering
  using Base

  export k_means, dp_means

  load("Clustering/src/types.jl")
  load("Clustering/src/k_means.jl")
  load("Clustering/src/dp_means.jl")
end
