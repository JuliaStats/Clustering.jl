module Clustering
    using Distances
    using NearestNeighbors
    using StatsBase

    using Printf
    using LinearAlgebra
    using SparseArrays
    using Statistics

    import Base: show
    import StatsBase: IntegerVector, RealVector, RealMatrix, counts

    export
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
    DbscanResult, DbscanCluster, dbscan,

    # fuzzy_cmeans
    fuzzy_cmeans, FuzzyCMeansResult,

    # counts
    counts, # reexport StatsBase.counts

    # silhouette
    silhouettes,

    # varinfo
    varinfo,

    # randindex
    randindex,

    # V-measure
    vmeasure,

    # hclust
    Hclust, hclust, cutree,

    # MCL
    mcl, MCLResult

    ## source files

    include("utils.jl")
    include("seeding.jl")

    include("kmeans.jl")
    include("kmedoids.jl")
    include("affprop.jl")
    include("dbscan.jl")
    include("mcl.jl")
    include("fuzzycmeans.jl")

    include("counts.jl")
    include("silhouette.jl")
    include("randindex.jl")
    include("varinfo.jl")
    include("vmeasure.jl")

    include("hclust.jl")

    include("deprecate.jl")
end
