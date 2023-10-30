# interface of hard clustering indices + general docs

"""
For hard clustering:

    clustering_quality(data, centers, assignments; quality_index, [metric])
    clustering_quality(data, clustering; quality_index, [metric])

For fuzzy clustering:

    clustering_quality(data, centers, weights; quality_index, fuzziness, [metric])
    clustering_quality(data, clustering; quality_index, fuzziness, [metric])
    
For hard clustering without cluster centers known:

    clustering_quality(assignments, dist_matrix; quality_index)
    clustering_quality(clustering, dist_matrix; quality_index)
    clustering_quality(data, assignments; quality_index, [metric])
    clustering_quality(data, clustering; quality_index, [metric])

Compute the clustering quality index for a given clustering.

Returns a real number which is the value of the chosen quality index type of the given clustering.

# Arguments
 - `data::AbstractMatrix`: ``d×n`` data matrix with each column representing one ``d``-dimensional data point
 - `centers::AbstractMatrix`: ``d×k`` matrix with cluster centers represented as columns
 - `assignments::AbstractVector{Int}`: ``n`` vector of point assignments (cluster indices)
 - `weights::AbstractMatrix`: ``n×k`` matrix with fuzzy clustering weights, `weights[i,j]` is the degree of membership of ``i``-th data point to ``j``-th cluster
 - `clustering::Union{ClusteringResult, FuzzyCMeansResult}`: the output of the clustering method
 - `quality_index::Symbol`: quality index to calculate; see below for the supported options
 - `dist_matrix::AbstractMatrix`: a ``n×n`` pairwise distance matrix; `dist_matrix[i,j]` is the distance between ``i``-th and ``j``-th points

 # Keyword arguments
 - `quality_index::Symbol`: quality index to calculate; see below for the supported options
 - `fuzziness::Real`: clustering fuzziness > 1
 - `metric::SemiMetric=SqEuclidean()`: `SemiMetric` object that defines the metric/distance/similarity function

When calling `clustering_quality` one can give `centers`, `assignments` or `weights` arguments by hand or provide a single `clustering` argument from which the necessary data will be read automatically.

For clustering without known cluster centers the datapoints are not required, only `dist_matrix` is necessary. If given, `data` and `metric` will be used to calculate distance matrix instead.

# Supported quality indices

Symbols ↑/↓ are quality direction.
- `:calinski_harabasz`: hard or fuzzy Calinski-Harabsz index (↑) returns the corrected ratio of between cluster centers inertia and within-clusters inertia
- `:xie_beni`: hard or fuzzy Xie-Beni index (↓) returns ratio betwen inertia within clusters and minimal distance between cluster centers
- `:davies_bouldin`: Davies-Bouldin index (↓) returns average similarity between each cluster and its most similar one, averaged over all the clusters
- `:dunn`: Dunn index (↑) returns ratio between minimal distance between clusters and maximal cluster diameter; it does not make use of `centers` argument
- `:silhouettes`: average silhouette index (↑), for all silhouettes use [`silhouettes`](@ref) method instead; it does not make use of `centers` argument
Please refer to the [documentation](@ref clustering_quality) for the definitions and usage descriptions of the supported quality indices. 

"""
function clustering_quality(
        data::AbstractMatrix{<:Real},           # d×n matrix
        centers::AbstractMatrix{<:Real},        # d×k matrix
        assignments::AbstractVector{<:Integer}; # n vector
        quality_index::Symbol,
        metric::SemiMetric=SqEuclidean()
    )
    d, n = size(data)
    dc, k = size(centers)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `data` and `centers`."))
    (1 <= k <= n) || throw(ArgumentError("Number of clusters k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    for (i, clu) in enumerate(assignments)
        (clu in axes(centers, 2)) || throw(ArgumentError("Invalid cluster index: assignments[$i]=$(clu)."))
    end
    for i in 1:k
        i ∉ assignments && @warn "Cluster number $(i) is empty. Clustering quality calculation may not be reliable."
    end

    if quality_index == :calinski_harabasz
        _cluquality_calinski_harabasz(metric, data, centers, assignments, nothing)
    elseif quality_index == :xie_beni
        _cluquality_xie_beni(metric, data, centers, assignments, nothing)
    elseif quality_index == :davies_bouldin
        _cluquality_davies_bouldin(metric, data, centers, assignments)
    elseif quality_index == :silhouettes
        mean(silhouettes(assignments, pairwise(metric, data, dims=2)))
    elseif quality_index == :dunn 
        _cluquality_dunn(assignments, pairwise(metric, data, dims=2))
    else
        throw(ArgumentError("Quality index $quality_index not supported."))
    end
end

clustering_quality(data::AbstractMatrix{<:Real}, R::KmeansResult; quality_index::Symbol, metric::SemiMetric=SqEuclidean()) =
    clustering_quality(data, R.centers, R.assignments; quality_index = quality_index, metric = metric)


# interface of fuzzy clustering indices

function clustering_quality(
        data::AbstractMatrix{<:Real},    # d×n matrix
        centers::AbstractMatrix{<:Real}, # d×k matrix
        weights::AbstractMatrix{<:Real}; # n×k matrix
        quality_index::Symbol,
        fuzziness::Real,
        metric::SemiMetric=SqEuclidean()
    )
    d, n = size(data)
    dc, k = size(centers)
    nw, kw = size(weights)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `data` and `centers`."))
    n == nw || throw(DimensionMismatch("Inconsistent data length for `data` and `weights`."))
    k == kw || throw(DimensionMismatch("Inconsistent number of clusters for `centers` and `weights`."))
    (1 <= k <= n) || throw(ArgumentError("Number of clusters k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    all(>=(0), weights) || throw(ArgumentError("All weights must be larger or equal 0."))
    1 < fuzziness || throw(ArgumentError("Fuzziness must be greater than 1 ($fuzziness given)"))

    if quality_index == :calinski_harabasz
        _cluquality_calinski_harabasz(metric, data, centers, weights, fuzziness)
    elseif quality_index == :xie_beni
        _cluquality_xie_beni(metric, data, centers, weights, fuzziness)
    else
        throw(ArgumentError("Quality index $quality_index not supported."))
    end
end

clustering_quality(data::AbstractMatrix{<:Real}, R::FuzzyCMeansResult; quality_index::Symbol, fuzziness::Real, metric::SemiMetric=SqEuclidean()) =
    clustering_quality(data, R.centers, R.weights; quality_index = quality_index, fuzziness = fuzziness, metric = metric)


# interface of clustering indices with cluster centres not known

function clustering_quality( 
        assignments::AbstractVector{<:Integer}, # n vector
        dist::AbstractMatrix{<:Real};           # n×n matrix
        quality_index::Symbol 
    )
    n, m = size(dist)
    na = length(assignments)
    n == m || throw(ArgumentError("Distance matrix must be square."))
    n == na || throw(DimensionMismatch("Inconsistent array dimensions for distance matrix and assignments."))

    if quality_index == :silhouettes
        mean(silhouettes(assignments, dist))
    elseif quality_index == :dunn
        _cluquality_dunn(assignments, dist)
    else
        throw(ArgumentError("Quality index $quality_index not supported."))
    end
end


clustering_quality(data::AbstractMatrix{<:Real}, assignments::AbstractVector{<:Integer}; quality_index::Symbol, metric::SemiMetric=SqEuclidean()) = 
    clustering_quality(assignments, pairwise(metric, data, dims=2); quality_index = quality_index)

clustering_quality(data::AbstractMatrix{<:Real}, R::ClusteringResult;  quality_index::Symbol, metric::SemiMetric=SqEuclidean()) =
    clustering_quality(R.assignments, pairwise(metric, data, dims=2); quality_index = quality_index)

clustering_quality(R::ClusteringResult, dist::AbstractMatrix{<:Real}; quality_index::Symbol) = 
    clustering_quality(R.assignments, dist; quality_index = quality_index)


# utility functions

function _gather_samples(assignments, k) # cluster_samples[j]: indices of points in cluster j
    cluster_samples = [Int[] for _ in  1:k]
    for (i, a) in zip(eachindex(assignments), assignments)
        push!(cluster_samples[a], i)
    end
    return cluster_samples
end

# shared between hard clustering calinski_harabasz and xie_beni
function _inner_inertia(
        metric::SemiMetric,
        data::AbstractMatrix,
        centers::AbstractMatrix,
        assignments::AbstractVector{<:Integer},
        fuzziness::Nothing
    ) 
    inner_inertia = sum(
        sum(colwise(metric, view(data, :, samples), center))
            for (center, samples) in zip(eachcol(centers), _gather_samples(assignments, size(centers)[2]))
    )
    return inner_inertia
end

# shared between fuzzy clustering calinski_harabasz and xie_beni (fuzzy version)
function _inner_inertia(
        metric::SemiMetric,
        data::AbstractMatrix,
        centers::AbstractMatrix,
        weights::AbstractMatrix,
        fuzziness::Real
    )
    pointCentreDistances = pairwise(metric, data, centers, dims=2)
    inner_inertia = sum(
        w^fuzziness * d for (w, d) in zip(weights, pointCentreDistances) 
    )
    return inner_inertia
end

# hard outer inertia for calinski_harabasz
function _outer_inertia(
        metric::SemiMetric,
        data::AbstractMatrix,
        centers::AbstractMatrix,
        assignments::AbstractVector{<:Integer},
        fuzziness::Nothing
    )
    global_center = vec(mean(data, dims=2))
    center_distances = colwise(metric, centers, global_center)
    return sum(center_distances[clu] for clu in assignments)
end

# fuzzy outer inertia for calinski_harabasz
function _outer_inertia(
        metric::SemiMetric,
        data::AbstractMatrix,
        centers::AbstractMatrix,
        weights::AbstractMatrix,
        fuzziness::Real
    )

    global_center = vec(mean(data, dims=2))
    center_distances = colwise(metric, centers, global_center)
    return sum(sum(w^fuzziness for w in view(weights, :, clu)) * d
                for (clu, d) in enumerate(center_distances)
            )
end

# Calinsk-Harabasz index
function  _cluquality_calinski_harabasz(
        metric::SemiMetric,
        data::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::Union{AbstractVector{<:Integer}, AbstractMatrix{<:Real}},
        fuzziness::Union{Real, Nothing}
    )
    n, k = size(data, 2), size(centers, 2)
    outer_inertia = _outer_inertia(metric, data, centers, assignments, fuzziness)
    inner_inertia = _inner_inertia(metric, data, centers, assignments, fuzziness)

    return (outer_inertia / inner_inertia) * (n - k) / (k - 1)
end


# Davies Bouldin index
function _cluquality_davies_bouldin(
        metric::SemiMetric,
        data::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
    )
    clu_idx = axes(centers, 2)
    clu_samples = _gather_samples(assignments, length(clu_idx))
    clu_diams = [mean(colwise(metric, view(data, :, samples), view(centers, :, clu)))
                 for (clu, samples) in zip(clu_idx, clu_samples)]
    center_dists = pairwise(metric, centers, dims=2)

    DB = mean(
        maximum(@inbounds (clu_diams[j₁] + clu_diams[j₂]) / center_dists[j₁,j₂] for j₂ in clu_idx if j₂ ≠ j₁)
            for j₁ in clu_idx
    )
    return  DB
end


# Xie-Beni index
function _cluquality_xie_beni(
        metric::SemiMetric,
        data::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::Union{AbstractVector{<:Integer}, AbstractMatrix{<:Real}},
        fuzziness::Union{Real, Nothing}
    )
    n, k = size(data, 2), size(centers,2)
    inner_intertia  = _inner_inertia(metric, data, centers, assignments, fuzziness)
    center_distances = pairwise(metric, centers, dims=2)
    min_center_distance = minimum(center_distances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k)

    return inner_intertia / (n * min_center_distance)
end

# Dunn index
function _cluquality_dunn(assignments::AbstractVector{<:Integer}, dist::AbstractMatrix{<:Real})

    max_inner_distance, min_outer_distance = typemin(eltype(dist)), typemax(eltype(dist))
    
    for i in eachindex(assignments), j in (i + 1):lastindex(assignments)
        @inbounds d = dist[i,j]
        if assignments[i] == assignments[j]
            if max_inner_distance < d
                max_inner_distance = d
            end
        else
            if min_outer_distance > d
                min_outer_distance = d
            end
        end
    end
    return min_outer_distance / max_inner_distance
end
