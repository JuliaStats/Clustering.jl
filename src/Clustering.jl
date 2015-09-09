module Clustering
    using ArrayViews
    using Distances
    using StatsBase
    
    import Base: show
    import StatsBase: IntegerVector, RealVector, RealMatrix, counts

    export

    # reexport from ArrayViews
    view,

    # reexport from StatsBase
    sample, sample!,

    # common
    ClusteringResult,
    nclusters, counts, assignments, 

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
    AffinityPropResult, affinityprop,

    # dbscan
    DbscanResult, dbscan,

    # silhouette
    silhouettes,

    # varinfo
    varinfo, 

    # randindex
    randindex,

    # hclust
    Hclust, hclust, cutree


    ## source files

    include("utils.jl")
    include("seeding.jl")

    include("kmeans.jl")
    include("kmedoids.jl")
    include("affprop.jl")
    include("dbscan.jl")

    include("silhouette.jl")
    include("randindex.jl")
    include("varinfo.jl")

    include("hclust.jl")

    include("deprecate.jl")
end
