# Silhouette

zero_tol(::Type{Float64})   = 1e-6
zero_tol(::Type{Float32})   = 1f-6
zero_tol(::Type{T}) where {T <: Integer} = zero(T)

# this function returns r of size (k, n), such that
# r[i, j] is the sum of distances of all points from cluster i to sample j
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
    silhouettes(assignments::AbstractVector, [counts,] dists)
    silhouettes(clustering::ClusteringResult, dists)

Compute silhouette values for individual points w.r.t. given clustering.

  * `assignments` the vector of point assignments (cluster indices)
  * `counts` the optional vector of cluster sizes (how many points assigned to each cluster; should match `assignments`)
  * `clustering` the output of some clustering method
  * `dists` point×point pairwise distance matrix

Returns a vector of silhouette values for each individual point.

`mean(silhouettes(...))` could be used as a measure of clustering quality;
higher values indicate better separation of clusters w.r.t. distances provided in `dists`.

#### References
  1. [Silhouette Wikipedia page](http://en.wikipedia.org/wiki/Silhouette_(clustering)).
  2. Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis". Computational and Applied Mathematics. 20: 53–65.
"""
function silhouettes(assignments::AbstractVector{<:Integer},
                     counts::AbstractVector{<:Integer},
                     dists::AbstractMatrix{T}) where T<:Real

    n = length(assignments)
    k = length(counts)
    for j = 1:n
        (1 <= assignments[j] <= k) || throw(ArgumentError("Bad assignments[$j]=$(assignments[j]): should be in 1:$k range."))
    end
    sum(counts) == n || throw(ArgumentError("Mismatch between assignments ($n) and counts (sum(counts)=$(sum(counts)))."))
    size(dists) == (n, n) || throw(DimensionMismatch("The size of a distance matrix ($(size(dists))) doesn't match the length of assignment vector ($n)."))

    # compute average distance from each cluster to each point --> r
    r = sil_aggregate_dists(k, assignments, dists)
    S = eltype(r)
    # from sum to average
    @inbounds for j = 1:n
        for i = 1:k
            c = counts[i]
            if i == assignments[j]
                c -= 1
            end
            if c == 0
                r[i,j] = zero(S)
            else
                r[i,j] /= c
            end
        end
    end
    # compute a and b
    # a: average distance w.r.t. the assigned cluster
    # b: the minimum average distance w.r.t. other cluster
    a = Vector{S}(undef, n)
    b = Vector{S}(undef, n)

    alleq = k == 1
    for j = 1:n
        l = assignments[j]
        a[j] = r[l, j]

        # When there is only one cluster all average distance from assigned
        # cluster have to be all equal. Otherwise, it's unlikely there will
        # be absolutely one cluster.
        alleq = alleq && abs(a[j] - a[1]) <= zero_tol(S)
        v = S(Inf)
        for i = 1:k
            @inbounds rij = r[i,j]
            i != l && rij < v && (v = rij)
        end
        b[j] = v 
    end
    # If there is only one cluster and distances are not all equal, they are
    # at zero distance from the other clusters. The clustering is invalid
    k == 1 && !alleq && fill!(b, zero(S))

    # compute silhouette score
    sil = a   # reuse the memory of a for sil
    for j = 1:n
        if counts[assignments[j]] == 1
            sil[j] = zero(S)
        else
            # b[j] and a[j] can be Inf so best to ensure Inf/Inf division
            # is avoided.
            @inbounds sil[j] = a[j] < b[j] ? one(S) - a[j]/b[j] :
                               a[j] > b[j] ? b[j]/a[j] - one(S) :
                               zero(S) 
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
