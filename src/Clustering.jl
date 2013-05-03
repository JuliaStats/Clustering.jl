module Clustering
    using Devectorize
    using Distance
    using MLBase

    import Base.show

    export k_means
    export kmeans, kmeans!, kmeans_opts, update!

    include("seeding.jl")
    include("kmeans.jl")
end
