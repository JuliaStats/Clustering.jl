module Clustering
    using Devectorize
    using Distance
    using Stats
    
    import Base.show

    export kmeans, kmeans!, kmeans_opts, update!

    export AffinityPropagationOpts
    export affinity_propagation

    include("seeding.jl")
    include("kmeans.jl")
    include("affprop.jl")
end
