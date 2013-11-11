module Clustering
    using NumericExtensions
    using Distance
    using Distributions
    
    import Base.show

    export kmeans, kmeans!, kmeans_opts, update!

    export AffinityPropagationOpts
    export affinity_propagation

    include("utils.jl")
    include("seeding.jl")
    include("kmeans.jl")
    include("affprop.jl")
    include("dbscan.jl")
end
