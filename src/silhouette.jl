# Silhouette

struct SqEuclideanPrecomputedSilhouettes{T}
    nclusters::Int
    dims::Int
    counts::Vector{Int} #[nclusters, 1]
    Ψ::Vector{T} #[nclusters], This represents the second moments of each cluster
    Y::Matrix{T} #[dims, nclusters], This represents the first moments of each cluster
    """
    SqEuclideanPrecomputedSilhouettes(::Type{T}, nclusters::Int, dims::Int)
    Precomputations container for [`silhouettes_precompute_batch!`](@ref).
    See also [`silhouettes`](@ref), [`silhouettes_precompute_batch!`](@ref)
    """
    SqEuclideanPrecomputedSilhouettes(::Type{T}, nclusters::Int, dims::Int) where T<:Real= new{T}(nclusters, dims, 
                                                                                                  zeros(Int, nclusters),
                                                                                                  zeros(T, nclusters), 
                                                                                                  zeros(T, dims, nclusters))
end

# this does the same as sil_aggregate_dists, but uses the method in "Distributed Silhouette Algorithm: Evaluating Clustering on Big Data"
# https://arxiv.org/abs/2303.14102
# this implementation uses the SqEuclidean distance only
"""
silhouettes_precompute_batch!(pre::SqEuclideanPrecomputedSilhouettes{T}, assignments::AbstractVector{Int}, x::AbstractMatrix{T}) where T<:Real
Include a batch of data and cluster assignments (x,a) in the precomputations for silhouettes.
This implementation supports only square Euclidean distances at present.
See also [`silhouettes`](@ref) [`SqEuclideanPrecomputedSilhouettes`](@ref)

# Examples:
```julia-repl
Julia> nclusters=10; d=3; bs=100; n=10000;
Julia> pre = SqEuclideanPrecomputedSilhouettes(Float32, nclusters, d);
Julia> x = reshape(collect(Float32, 1:30000), 3, n); # direct computation is impractical on such a big dataset
Julia> a = reshape(repeat(collect(1:nclusters),trunc(Int, n/nclusters)), n);
Julia> batches_x = eachslice(reshape(x, 3, trunc(Int, n/bs), bs); dims=3); batches_a = eachslice(reshape(a, trunc(Int, n/bs), bs); dims=2);
Julia> @time [silhouettes_precompute_batch!(nclusters, aa, xx, pre) for (xx, aa) in zip(batches_x, batches_a)]; # precompute vectors on big data
0.838358 seconds (2.06 M allocations: 141.495 MiB, 4.24% gc time, 99.00% compilation time)
Julia> @time sil = vcat([silhouettes(xx, aa, pre) for (xx, aa) in zip(batches_x, batches_a)]); # calculate silhouette scores in batched fashion
0.049759 seconds (53.78 nclusters allocations: 4.440 MiB, 98.54% compilation time)
Julia> size(sil)
(10000,)
```
"""
function silhouettes_precompute_batch!(pre::SqEuclideanPrecomputedSilhouettes{T}, assignments::AbstractVector{Int}, x::AbstractMatrix{T}) where T<:Real
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

silhouettes_precompute_batch!(pre::SqEuclideanPrecomputedSilhouettes{T}, 
                              R::ClusteringResult, 
                              x::AbstractMatrix{T}) where T<:Real = silhouettes_precompute_batch!(pre, assignments(R), x)

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
    nclusters >= 2 || throw(ArgumentError("silhouettes() not defined for the degenerated clustering with a single cluster."))
    check_assignments(assignments, nclusters)
    sum(counts) == n || throw(ArgumentError("Mismatch between assignments ($n) and counts (sum(counts)=$(sum(counts)))."))
    size(dists) == (n, n) || throw(DimensionMismatch("The size of a distance matrix ($(size(dists))) doesn't match the length of assignment vector ($n)."))

    # compute average distance from each cluster to each point --> r
    cluster_to_point = sil_aggregate_dists(nclusters, assignments, dists)
    cluster_to_point = normalize_aggregate_distances!(cluster_to_point, assignments, counts)
    return cluster_to_point
end

function sil_aggregate_distances_normalized_streaming(x::AbstractMatrix{T}, assignments::AbstractVector{Int}, pre::SqEuclideanPrecomputedSilhouettes{T}) where T <: Real
    nclusters = pre.nclusters
    n = size(x, 2)
    nclusters >= 2 || throw(ArgumentError("silhouettes() not defined for the degenerated clustering with a single cluster."))
    size(x, 1) == size(pre.Y, 1) || throw(ArgumentError("input features dimension does not match with precomputation on dimension 1"))

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
    silhouettes(x::AbstractMatrix, assignments::AbstractVector, pre::SqEuclideanPrecomputedSilhouettes) -> Vector{Float64}

Compute *silhouette* values for individual points w.r.t. given clustering.

Returns the ``n``-length vector of silhouette values for each individual point.

# Arguments
 - `assignments::AbstractVector{Int}`: the vector of point assignments
   (cluster indices)
 - `counts::AbstractVector{Int}`: the optional vector of cluster sizes (how many
   points assigned to each cluster; should match `assignments`)
 - `clustering::ClusteringResult`: the output of some clustering method
 - `dists::AbstractMatrix`: ``n×n`` matrix of pairwise distances between
   the points
 - `x::AbstractMatrix`: `d×n`` matrix of ``n`` data features of dimensionality ``d``
 - `pre::SqEuclideanPrecomputedSilhouettes`: precomputed vectors of cluster silhouettes.

# References
> Peter J. Rousseeuw (1987). *Silhouettes: a Graphical Aid to the
> Interpretation and Validation of Cluster Analysis*. Computational and
> Applied Mathematics. 20: 53–65.
> Marco Gaido (2023). Distributed Silhouette Algorithm: Evaluating Clustering on Big Data 
"""
function silhouettes(assignments::AbstractVector{<:Integer},
                     counts::AbstractVector{<:Integer},
                     dists::AbstractMatrix{T}) where T<:Real

    cluster_to_point = sil_aggregate_distances_normalized(assignments, counts, dists)
    return silhouette_scores(cluster_to_point, assignments, counts)
end

function silhouettes(x::AbstractMatrix{T}, assignments::AbstractVector{<:Integer}, pre::SqEuclideanPrecomputedSilhouettes{T}) where T<:Real
    cluster_to_point = sil_aggregate_distances_normalized_streaming(x, assignments, pre)
    return silhouette_scores(cluster_to_point, assignments, pre.counts)
end

silhouettes(R::ClusteringResult, dists::AbstractMatrix) =
    silhouettes(assignments(R), counts(R), dists)

function silhouettes(assignments::AbstractVector{<:Integer}, dists::AbstractMatrix)
    counts = fill(0, maximum(assignments))
    for a in assignments
        counts[a] += 1
    end
    silhouettes(assignments, counts, dists)
end
