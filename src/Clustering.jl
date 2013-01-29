require("options")

module Clustering
    using Distributions
    using OptionsMod

    import Base.show

    export k_means, dp_means

    include("types.jl")
    include("k_means.jl")
    include("dp_means.jl")
end
