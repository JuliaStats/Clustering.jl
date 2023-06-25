# Silhouette

mutable struct SilhouettesDistsPrecompute{T}
    k::Int
    d::Int
    counts::Matrix{Int} #[k, 1]
    Ψ::Matrix{T} #[k, 1]
    Y::Matrix{T} #[d, k]
end

SilhouettesDistsPrecompute(k::Int, d::Int, ::Type{T}) where T<:Real= SilhouettesDistsPrecompute{T}(k, d, 
                                                                                      zeros(Int, k, 1),
                                                                                      zeros(T, k, 1), 
                                                                                      zeros(T, d, k))

                                                                                      # this does the same as sil_aggregate_dists, but uses the method in "Distributed Silhouette Algorithm: Evaluating Clustering on Big Data"
# https://arxiv.org/abs/2303.14102
# this implementation uses the square
function silhouettes_precompute_batch!(k::Int, a::AbstractVector{Int}, x::AbstractMatrix{T}, pre::SilhouettesDistsPrecompute{T}) where T<:Real
    # x dims are [D,N]
    d, n = size(x)
    all(1 .<= a .<= k) || throw(ArgumentError("Bad assignments: should be in 1:$k range."))
    d == pre.d || throw(ArgumentError("Bad data: x[1]=$d must match pre.d=$(pre.d)."))
    # update counts
    for (val, cnt) in countmap(a)
        pre.counts[val] += cnt
    end
    @assert n == length(a) "data matrix dimensions must be (..., $(length(a))) and got $(size(x))"
    ξ = sum(x.^2; dims=1) # [1,n]
    # precompute vectors
    for kk in 1:k
        @inbounds pre.Y[:, kk] += sum((reshape(a, 1, n) .== kk) .* x; dims=2)
        @inbounds pre.Ψ[kk] += sum((reshape(a, 1, n) .== kk) .* ξ)
    end
end

function silhouettes_precompute_batch!(R::ClusteringResult, 
                                              x::AbstractMatrix{T}, 
                                              pre::SilhouettesDistsPrecompute{T}) where T<:Real
    silhouettes_precompute_batch!(nclusters(R), assignments(R), x, pre)
end

# this function returns r of size (k, n), such that
# r[i, j] is the sum of distances of all points from cluster i to point j
#
function sil_aggregate_dists(k::Int, a::AbstractVector{Int}, dists::AbstractMatrix{T}) where T<:Real
    n = length(a)
    S = typeof((one(T)+one(T))/2)
    r = zeros(S, k, n)
    @inbounds for j = 1:n
        for i = 1:j-1
            r[a[i],j] += dists[i,j]
        end
        for i = j+1:n
            r[a[i],j] += dists[i,j]
        end
    end
    return r
end

function normalize_aggregate_dists!(r, assignments, counts, k, n)
    # from sum to average
    @inbounds for j = 1:n
        for i = 1:k
            c = copy(counts[i])
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

function sil_aggregate_dists_normalized(assignments::AbstractVector{<:Integer},
                                        counts::AbstractVector{<:Integer},
                                        dists::AbstractMatrix{T}) where T<:Real
    n = length(assignments)
    k = length(counts)
    k >= 2 || throw(ArgumentError("silhouettes() not defined for the degenerated clustering with a single cluster."))
    for j = 1:n
        (1 <= assignments[j] <= k) || throw(ArgumentError("Bad assignments[$j]=$(assignments[j]): should be in 1:$k range."))
    end
    sum(counts) == n || throw(ArgumentError("Mismatch between assignments ($n) and counts (sum(counts)=$(sum(counts)))."))
    size(dists) == (n, n) || throw(DimensionMismatch("The size of a distance matrix ($(size(dists))) doesn't match the length of assignment vector ($n)."))

    # compute average distance from each cluster to each point --> r
    r = sil_aggregate_dists(k, assignments, dists)
    r = normalize_aggregate_dists!(r, assignments, counts, k, n)
    return r
end

function sil_aggregate_dists_normalized_streaming(x::AbstractMatrix{T}, assignments::AbstractVector{Int}, pre::SilhouettesDistsPrecompute{T}) where T <: Real
    k = pre.k
    n = size(x, 2)
    k >= 2 || throw(ArgumentError("silhouettes() not defined for the degenerated clustering with a single cluster."))
    size(x, 1) == size(pre.Y, 1) || throw(ArgumentError("input features dimension does not match with precomputation on dimension 1"))

    # compute average distance from each cluster to each point --> r
    ξx = sum(x.^2; dims=1) # [1,n]
    r = pre.counts .* ξx .+ pre.Ψ .- 2 .* transpose(pre.Y) * x # r is [k, n]
    r = normalize_aggregate_dists!(r, assignments, pre.counts, k, n)
    return r
end

function silhouettes_given_dist(r::Matrix{T}, assignments::AbstractVector{Int}, counts::Vector{Int}) where {T}
    n = length(assignments)
    # compute a and b
    # a: average distance w.r.t. the assigned cluster
    # b: the minimum average distance w.r.t. other cluster
    a = similar(r, n)
    b = similar(r, n)
    k = length(counts)
    for j = 1:n
        l = assignments[j]
        a[j] = r[l, j]

        v = typemax(eltype(b))
        @inbounds for i = 1:k
            counts[i] == 0 && continue # skip empty clusters
            rij = r[i,j]
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
                               zero(eltype(r))
        end
    end
    return sil
end


"""
    silhouettes(assignments::AbstractVector, [counts,] dists) -> Vector{Float64}
    silhouettes(clustering::ClusteringResult, dists) -> Vector{Float64}
    silhouettes(x::AbstractMatrix, assignments::AbstractVector, pre::SilhouettesDistsPrecompute) -> Vector{Float64}

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

# References
> Peter J. Rousseeuw (1987). *Silhouettes: a Graphical Aid to the
> Interpretation and Validation of Cluster Analysis*. Computational and
> Applied Mathematics. 20: 53–65.
"""
function silhouettes(assignments::AbstractVector{<:Integer},
                     counts::AbstractVector{<:Integer},
                     dists::AbstractMatrix{T}) where T<:Real

    r = sil_aggregate_dists_normalized(assignments, counts, dists)
    return silhouettes_given_dist(r, assignments, counts)
end

function silhouettes(x::AbstractMatrix{T}, assignments::AbstractVector{<:Integer}, pre::SilhouettesDistsPrecompute{T}) where T<:Real
    r = sil_aggregate_dists_normalized_streaming(x, assignments, pre)
    return silhouettes_given_dist(r, assignments, reshape(pre.counts, :))
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
