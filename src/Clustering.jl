load("DataFrames")

module Clustering
  using Base
  using DataFrames

  import Base.print, Base.show, Base.repl_show

  export k_means, dp_means

  load("Clustering/src/types.jl")
  load("Clustering/src/k_means.jl")
  load("Clustering/src/dp_means.jl")
end
