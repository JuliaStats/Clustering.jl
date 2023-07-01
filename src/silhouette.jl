# Silhouette
"""
Base type for efficient computation of [`silhouettes`](@ref) clustering metric.
"""
abstract type SilhouettesCache{T} end

struct SqEuclideanSilhouettesCache{T} <: SilhouettesCache{T}
    nclusters::Int
    dims::Int
    counts::Vector{Int} #[nclusters, 1]
    Ψ::Vector{T} #[nclusters], This represents the second moments of each cluster
    Y::Matrix{T} #[dims, nclusters], This represents the first moments of each cluster
    """
    SqEuclideanSilhouettesCache(::Type{T}, nclusters::Int, dims::Int)
    Precomputations container for [`precompute_silhouettes`](@ref).
    See also [`silhouettes`](@ref), [`precompute_silhouettes`](@ref)
    """
    SqEuclideanSilhouettesCache(::Type{T}, nclusters::Int, dims::Int) where T<:Real= new{T}(nclusters, dims, 
                                                                                                  zeros(Int, nclusters),
                                                                                                  zeros(T, nclusters), 
                                                                                                  zeros(T, dims, nclusters))
end

"""
silhouettes_cache(t::Type, metric::Union{Metric, SemiMetric}, nclusters::Int, dims::Int)
Return an SilhouettesCache object which can be used to process large amounts of data to get silhouette scores.
This implementation only supports square Euclidean distances at present.
See also [`silhouettes`](@ref) [`SqEuclideanSilhouettesCache`](@ref)

# Examples:
```julia-repl
Julia> using Distances
Julia> nclusters=10; dims=3; bs=100; n=10000;
Julia> pre = silhouettes_cache(Float32, SqEuclidean(), nclusters, dims);
Julia> x = reshape(collect(Float32, 1:30000), dims, n); # direct computation is impractical on such a big dataset
Julia> a = reshape(repeat(collect(1:nclusters),trunc(Int, n/nclusters)), n);
Julia> batches_x = eachslice(reshape(x, dims, trunc(Int, n/bs), bs); dims=3); batches_a = eachslice(reshape(a, trunc(Int, n/bs), bs); dims=2);
Julia> @time [pre(xx, aa) for (xx, aa) in zip(batches_x, batches_a)]; # precompute vectors on big data
0.502363 seconds (956.08 k allocations: 65.143 MiB, 2.23% gc time, 99.85% compilation time)
Julia> @time sil = vcat([silhouettes(xx, aa, pre) for (xx, aa) in zip(batches_x, batches_a)]...); # calculate silhouette scores in batched fashion
0.046788 seconds (54.28 k allocations: 4.346 MiB, 98.45% compilation time)
Julia> size(sil)
(10000,)
```
"""
silhouettes_cache(::Type, metric::Union{Metric, SemiMetric}, nclusters, dims) = error("precomputed_silhouettes() for metric $(typeof(metric)) is not implemented")
silhouettes_cache(t::Type, metric::SqEuclidean, nclusters::Int, dims::Int)::SilhouettesCache = SqEuclideanSilhouettesCache(t, nclusters, dims)

# this does the same as sil_aggregate_dists, but uses the method in "Distributed Silhouette Algorithm: Evaluating Clustering on Big Data"
# https://arxiv.org/abs/2303.14102
# this implementation uses the SqEuclidean distance only
function precompute_silhouettes(pre::SqEuclideanSilhouettesCache{T}, x::AbstractMatrix{T}, assignments::AbstractVector{Int}) where T<:Real
    # x dims are [D,N]
    d, n = size(x)
    check_assignments(assignments, pre.nclusters)
    d == pre.dims || throw(ArgumentError("Bad data: x[1]=$d must match pre.d=$(pre.d)."))
    # update counts
    for (val, cnt) in countmap(assignments)
        pre.counts[val] += cnt
    end
    @assert n == length(assignments) "data matrix dimensions must be (..., $(length(assignments))) and got $(size(x))"
    ξ = sum(abs2, x; dims=1) # [1,n]
    # precompute vectors
    @inbounds for (i, cluster) in enumerate(assignments)
        pre.Y[:, cluster] .+= view(x, :, i)
        pre.Ψ[cluster] += ξ[i]
    end
    return pre
end

precompute_silhouettes(pre::SqEuclideanSilhouettesCache{T}, 
                       x::AbstractMatrix{T}, R::ClusteringResult) where T<:Real = precompute_silhouettes(pre, x, assignments(R))


function (pre::SqEuclideanSilhouettesCache)(x::AbstractMatrix{T}, assignments::Union{AbstractVector{Int}, ClusteringResult}) where T<:Real
    precompute_silhouettes(pre, x, assignments)
end

# this function returns r of size (nclusters, n), such that
# r[i, j] is the sum of distances of all points from cluster i to point j
#
function sil_aggregate_dists(nclusters::Int, assignments::AbstractVector{Int}, dists::AbstractMatrix{T}) where T<:Real
    n = length(assignments)
    S = typeof((one(T)+one(T))/2)
    r = zeros(S, nclusters, n)
    @inbounds for j = 1:n
        for i = 1:j-1
            r[assignments[i],j] += dists[i,j]
        end
        for i = j+1:n
            r[assignments[i],j] += dists[i,j]
        end
    end
    return r
end

function normalize_aggregate_distances!(r, assignments, counts)
    # from sum to average
    @assert size(r) == (length(counts), length(assignments)) "silhouettes: cluster_to_point matrix dimensions $(size(r)) do not match assignments and counts - $(((length(counts), length(assignments))))"
    @inbounds for j in eachindex(assignments)
        for i in eachindex(counts)
            c = counts[i]
            if i == assignments[j]
                c -= 1
            end
            if c == 0
                r[i,j] = 0
            else
                r[i,j] /= c
            end
        end
    end
    return r
end

function sil_aggregate_distances_normalized(assignments::AbstractVector{<:Integer},
                                        counts::AbstractVector{<:Integer},
                                        dists::AbstractMatrix{T}) where T<:Real
    n = length(assignments)
    nclusters = length(counts)
    check_assignments(assignments, nclusters)
    sum(counts) == n || throw(ArgumentError("Mismatch between assignments ($n) and counts (sum(counts)=$(sum(counts)))."))
    size(dists) == (n, n) || throw(DimensionMismatch("The size of a distance matrix ($(size(dists))) doesn't match the length of assignment vector ($n)."))

    # compute average distance from each cluster to each point --> r
    cluster_to_point = sil_aggregate_dists(nclusters, assignments, dists)
    cluster_to_point = normalize_aggregate_distances!(cluster_to_point, assignments, counts)
    return cluster_to_point
end

function sil_aggregate_distances_normalized_streaming(x::AbstractMatrix{T}, assignments::AbstractVector{Int}, pre::SqEuclideanSilhouettesCache{T}) where T <: Real
    nclusters = pre.nclusters
    check_assignments(assignments, nclusters)
    size(x, 1) == size(pre.Y, 1) || throw(ArgumentError("silhouette: number of input features (dimension 1) used in pre computation was $(size(pre.Y, 1)), got $(size(x, 1))"))
    size(x, 2) == length(assignments) || throw(ArgumentError("silhouette: number of input points $(size(x, 2)) do not match number of assignments $(length(assignments))"))

    # compute average distance from each cluster to each point --> r
    ξx = sum(abs2, x; dims=1) # [1,n]
    cluster_to_point = reshape(pre.counts, :, 1) .* ξx .+ reshape(pre.Ψ, pre.nclusters, 1) .- 2 .* transpose(pre.Y) * x # cluster_to_point is [nclusters, n]
    cluster_to_point = normalize_aggregate_distances!(cluster_to_point, assignments, pre.counts)
    return cluster_to_point
end

# compute silhouette scores for each point given a matrix of cluster-to-point distances
function silhouette_scores(cluster_to_point::AbstractMatrix{<:Real}, assignments::AbstractVector{Int}, counts::Vector{Int})
    n = length(assignments)
    # compute a and b
    # a: average distance w.r.t. the assigned cluster
    # b: the minimum average distance w.r.t. other cluster
    a = similar(cluster_to_point, n)
    b = similar(cluster_to_point, n)
    nclusters = length(counts)
    for j = 1:n
        l = assignments[j]
        a[j] = cluster_to_point[l, j]

        v = typemax(eltype(b))
        @inbounds for i = 1:nclusters
            counts[i] == 0 && continue # skip empty clusters
            rij = cluster_to_point[i,j]
            if (i != l) && (rij < v)
                v = rij
            end
        end
        b[j] = v
    end

    # compute silhouette score
    sil = a   # reuse the memory of a for sil
    for j = 1:n
        if counts[assignments[j]] == 1
            sil[j] = 0
        else
            #If both a[i] and b[i] are equal to 0 or Inf, silhouettes is defined as 0
            @inbounds sil[j] = a[j] < b[j] ? 1 - a[j]/b[j] :
                               a[j] > b[j] ? b[j]/a[j] - 1 :
                               zero(eltype(cluster_to_point))
        end
    end
    return sil
end


"""
    silhouettes(assignments::AbstractVector, [counts,] dists) -> Vector{Float64}
    silhouettes(clustering::ClusteringResult, dists) -> Vector{Float64}
    silhouettes(x::AbstractMatrix, assignments::AbstractVector, pre::SilhouettesCache{T}) -> Vector{Float64}
    silhouettes(metric::Union{Metric, SemiMetric}, assignments::AbstractVector{<:Integer}, x::AbstractMatrix; 
                nclusters=maximum(assignments), batch_size=size(x, 2), pre_calculate::Bool=true)

Compute *silhouette* values for individual points w.r.t. given clustering.

Returns the ``n``-length vector of silhouette values for each individual point.

# Arguments
 - `assignments::Union{AbstractVector{Int}, ClusteringResult}`: the vector of point assignments
   (cluster indices)
 - `points::AbstractMatrix`: if metric is nothing it is an ``n×n`` matrix of pairwise distances between the points,
   otherwise it is an ``d×n`` matrix of `d` dimensional clustered data points.
 - `metric::Union{Metric, SemiMetric, Nothing}`: an instance of Distances Metric object or nothing, 
   indicating the distance metric used for calculating point distances.
 - `counts::AbstractVector{Int}`: the optional vector of cluster sizes (how many
   points assigned to each cluster; should match `assignments`)
 - `nclusters` optional, how many clusters are associated with the assignments (recommended to provide when distances are not provided).
 - `method` optional, one of `[:default, :classis, :cached]`. controls use of cached algorithm.
   :default selects the best supported algorithm given the metric.

# References
> Peter J. Rousseeuw (1987). *Silhouettes: a Graphical Aid to the
> Interpretation and Validation of Cluster Analysis*. Computational and
> Applied Mathematics. 20: 53–65.
> Marco Gaido (2023). Distributed Silhouette Algorithm: Evaluating Clustering on Big Data 
"""
function silhouettes(assignments::AbstractVector{<:Integer}, points:: AbstractMatrix;
                     metric::Union{Metric, SemiMetric, Nothing}=nothing,
                     counts::Union{AbstractVector{<:Integer}, Nothing}=nothing,
                     method::Symbol = :default,
                     nclusters=maximum(assignments))

    @assert in(method, [:default, :classic, :cached]) "method key argument must be one of :default, :classic or :cached"
    metric === nothing && @assert(iszero(diag(points)) && issymmetric(points), "silhouettes(): metric not provided, yet points matrix does not seem to be a distances matrix - symmetric and zero on the diagonal.")
    
    if method == :default
        if metric isa SqEuclidean
            method = :cached
        else
            method = :classic
        end
    end

    if method == :classic
        if counts === nothing
            counts = fill(0, maximum(assignments))
            for a in assignments
                counts[a] += 1
            end
        end
        if metric !== nothing
            dists = pairwise(metric, points, points; dims=2)
        else
            dists = points
        end
        
        cluster_to_point = sil_aggregate_distances_normalized(assignments, counts, dists)
        sil = silhouette_scores(cluster_to_point, assignments, counts)

    elseif  method == :cached
        !(metric isa SqEuclidean) && throw(ArgumentError("method :cached currently only supports SqEuclidean metric."))
        batch_size=size(points, 2) # a single batch. May not be optimal. Use `silhouettes_cache` and `silhouettes_using_cache` to change this.
        dims, n = size(points)
        check_assignments(assignments, nclusters)
        pre_calculate::SilhouettesCache = silhouettes_cache(eltype(points), metric, nclusters, dims)
        batched_x = eachslice(reshape(points, dims, batch_size, trunc(Int, n/batch_size)), dims=3)
        batched_assignments = eachslice(reshape(assignments, batch_size, trunc(Int, n/batch_size)), dims=2)
        [pre_calculate(x_batch, a_batch) for (x_batch, a_batch) in zip(batched_x, batched_assignments)] # run through batches of data and update cache
        silhouettes_streaming = []
        for (x_batch, a_batch) in zip(batched_x, batched_assignments)
            silhouettes_streaming = vcat(silhouettes_streaming, silhouettes_using_cache(x_batch, a_batch, pre_calculate)) # calculate each batch using the cache
        end
        sil = silhouettes_streaming
    end
    return sil
end

silhouettes(R::ClusteringResult, points:: AbstractMatrix;
            metric::Union{Metric, SemiMetric, Nothing}=nothing,
            counts::Union{AbstractVector{<:Integer}, Nothing}=nothing,
            method::Symbol = :default,
            nclusters=maximum(assignments))= silhouettes(assignments(R), points; metric=metric, counts=counts(R), method=method, nclusters=nclusters)

function silhouettes_using_cache(x::AbstractMatrix{T}, assignments::AbstractVector{<:Integer}, pre::SilhouettesCache{T}) where T<:Real
    cluster_to_point = sil_aggregate_distances_normalized_streaming(x, assignments, pre)
    return silhouette_scores(cluster_to_point, assignments, pre.counts)
end