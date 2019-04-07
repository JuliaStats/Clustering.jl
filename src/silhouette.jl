# Silhouette

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


"""
    silhouettes(assignments::AbstractVector, [counts,] dists) -> Vector{Float64}
    silhouettes(clustering::ClusteringResult, dists) -> Vector{Float64}

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
    # from sum to average
    @inbounds for j = 1:n
        for i = 1:k
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

    # compute a and b
    # a: average distance w.r.t. the assigned cluster
    # b: the minimum average distance w.r.t. other cluster
    a = similar(r, n)
    b = similar(r, n)

    for j = 1:n
        l = assignments[j]
        a[j] = r[l, j]

        v = typemax(eltype(b))
        for i = 1:k
            @inbounds rij = r[i,j]
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

silhouettes(R::ClusteringResult, dists::AbstractMatrix) =
    silhouettes(assignments(R), counts(R), dists)

function silhouettes(assignments::AbstractVector{<:Integer}, dists::AbstractMatrix)
    counts = fill(0, maximum(assignments))
    for a in assignments
        counts[a] += 1
    end
    silhouettes(assignments, counts, dists)
end
