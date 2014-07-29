module Clustering
    using ArrayViews
    using Distance
    using StatsBase
    
    import Base: show
    import StatsBase: IntegerVector, RealVector, RealMatrix

    export

    # reexport from ArrayViews
    view,

    # reexport from StatsBase
    sample, sample!,


    # seeding
    SeedingAlgorithm, 
    RandSeedAlg, KmppAlg, KmCentralityAlg,
    copyseeds, copyseeds!,
    initseeds, initseeds!, 
    initseeds_by_costs, initseeds_by_costs!,
    kmpp, kmpp_by_costs,

    # kmeans
    kmeans, kmeans!, KmeansResult, kmeans_opts, 

    # kmedoids
    kmedoids, kmedoids!, KmedoidsResult,

    # affprop
    AffinityPropagationOpts, affinity_propagation,

    # sil
    silhouettes


    ## source files

    include("utils.jl")
    include("seeding.jl")
    include("kmeans.jl")
    include("kmedoids.jl")
    include("affprop.jl")
    include("dbscan.jl")
    include("sil.jl")
    include("deprecate.jl")
end
