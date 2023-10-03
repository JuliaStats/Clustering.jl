# Silhouette

# compute silhouette scores for each point given a matrix of cluster-to-point distances
function silhouettes_scores(clu_to_pt::AbstractMatrix{<:Real},
                            assignments::AbstractVector{<:Integer},
                            clu_sizes::AbstractVector{<:Integer})
    n = length(assignments)
    @assert size(clu_to_pt) == (length(clu_sizes), n)

    # compute a and b
    # a: average distance w.r.t. the assigned cluster
    # b: the minimum average distance w.r.t. other cluster
    a = similar(clu_to_pt, n)
    b = similar(clu_to_pt, n)
    nclusters = length(clu_sizes)
    for j in 1:n
        l = assignments[j]
        a[j] = clu_to_pt[l, j]

        v = typemax(eltype(b))
        @inbounds for i = 1:nclusters
            clu_sizes[i] == 0 && continue # skip empty clusters
            rij = clu_to_pt[i, j]
            if (i != l) && (rij < v)
                v = rij
            end
        end
        b[j] = v
    end

    # compute silhouette score
    sil = a   # reuse the memory of a for sil
    for j = 1:n
        if clu_sizes[assignments[j]] == 1
            sil[j] = 0
        else
            #If both a[i] and b[i] are equal to 0 or Inf, silhouettes is defined as 0
            @inbounds sil[j] = a[j] < b[j] ? 1 - a[j]/b[j] :
                               a[j] > b[j] ? b[j]/a[j] - 1 :
                               zero(eltype(sil))
        end
    end
    return sil
end

# calculate silhouette scores (single batch)
silhouettes_batch(dists::ClusterDistances,
                  assignments::AbstractVector{<:Integer},
                  points::Union{AbstractMatrix, Nothing},
                  indices::AbstractVector{<:Integer}) =
    silhouettes_scores(meandistances(dists, assignments, points, indices),
                       assignments, cluster_sizes(dists))

# batch-calculate silhouette scores (splitting the points into chunks of batch_size size)
function silhouettes(dists::ClusterDistances,
                     assignments::AbstractVector{<:Integer},
                     points::Union{AbstractMatrix, Nothing},
                     batch_size::Union{Integer, Nothing} = nothing)
    n = length(assignments)
    ((batch_size === nothing) || (n <= batch_size)) &&
        return silhouettes_batch(dists, assignments, points, eachindex(assignments))

    return mapreduce(vcat, 1:batch_size:n) do batch_start
        batch_ixs = batch_start:min(batch_start + batch_size - 1, n)
        # copy points/assignments to speed up matrix and indexing operations
        silhouettes_batch(dists, assignments[batch_ixs],
                          points !== nothing ? points[:, batch_ixs] : nothing,
                          batch_ixs)
    end
end

"""
    silhouettes(assignments::Union{AbstractVector, ClusteringResult}, point_dists::Matrix) -> Vector{Float64}
    silhouettes(assignments::Union{AbstractVector, ClusteringResult}, points::Matrix;
                metric::SemiMetric, [batch_size::Integer]) -> Vector{Float64}

Compute *silhouette* values for individual points w.r.t. given clustering.

Returns the ``n``-length vector of silhouette values for each individual point.

# Arguments
 - `assignments::Union{AbstractVector{Int}, ClusteringResult}`: the vector of point assignments
   (cluster indices)
 - `points::AbstractMatrix`: if metric is nothing it is an ``n×n`` matrix of pairwise distances between the points,
   otherwise it is an ``d×n`` matrix of `d` dimensional clustered data points.
 - `metric::Union{SemiMetric, Nothing}`: an instance of Distances Metric object or nothing,
   indicating the distance metric used for calculating point distances.
 - `batch_size::Union{Integer, Nothing}`: if integer is given, calculate silhouettes in batches
   of `batch_size` points each, throws `DimensionMismatch` if batched calculation
   is not supported by given `metric`.

# References
> Peter J. Rousseeuw (1987). *Silhouettes: a Graphical Aid to the
> Interpretation and Validation of Cluster Analysis*. Computational and
> Applied Mathematics. 20: 53–65.
> Marco Gaido (2023). Distributed Silhouette Algorithm: Evaluating Clustering on Big Data
"""
function silhouettes(assignments::AbstractVector{<:Integer},
                     points::AbstractMatrix;
                     metric::Union{SemiMetric, Nothing} = nothing,
                     batch_size::Union{Integer, Nothing} = nothing)
    nclusters = maximum(assignments)
    nclusters >= 2 || throw(ArgumentError("silhouettes() not defined for the degenerated clustering with a single cluster."))
    check_assignments(assignments, nclusters)
    return silhouettes(ClusterDistances(metric, assignments, points, batch_size),
                       assignments, points, batch_size)
end

silhouettes(R::ClusteringResult, points::AbstractMatrix; kwargs...) =
    silhouettes(assignments(R), points; kwargs...)
