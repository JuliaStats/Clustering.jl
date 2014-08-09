# Silhouette


# this function returns r of size (k, n), such that
# r[i, j] is the sum of distances of all points from cluster i to sample j
#
function sil_aggregate_dists{T<:Real}(k::Int, a::AbstractVector{Int}, dists::DenseMatrix{T})
    n = length(a)
    r = zeros(k, n)
    for j = 1:n
        (1 <= a[j] <= k) || error("a[j] should have 1 <= a[j] <= k.")
    end
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


function silhouettes{T<:Real}(assignments::Vector{Int}, 
                              counts::AbstractVector{Int}, 
                              dists::DenseMatrix{T})

    n = length(assignments)
    k = length(counts)
    size(dists) == (n, n) || throw(DimensionMismatch("Inconsistent array dimensions."))

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
                r[i,j] = 0.0
            else
                r[i,j] /= c
            end
        end
    end

    # compute a and b
    # a: average distance w.r.t. the assigned cluster
    # b: the minimum average distance w.r.t. other cluster
    a = Array(Float64, n)
    b = Array(Float64, n)

    for j = 1:n
        l = assignments[j]
        a[j] = r[l, j]

        v = Inf
        p = -1
        for i = 1:k
            @inbounds rij = r[i,j]
            if (i != l) && (rij < v)
                v = rij
                p = i
            end
        end
        b[j] = v
    end

    # compute silhouette score 
    sil = a   # reuse the memory of a for sil
    for j = 1:n
        @inbounds sil[j] = (b[j] - a[j]) / max(a[j], b[j])
    end
    return sil
end

silhouettes(R::ClusteringResult, dists::DenseMatrix) = 
    silhouettes(assignments(R), counts(R), dists) 
