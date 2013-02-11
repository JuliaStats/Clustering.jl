require("Options")

module Clustering
    using Distributions
    using OptionsMod
	using MLBase

    # import Base.show

    export k_means, dp_means
	export kmeans, kmeans!, kmeans_opts, update!

    include("types.jl")
    include("k_means.jl")
    include("dp_means.jl")
	
	include("seeding.jl")
	include("kmeans.jl")
end
