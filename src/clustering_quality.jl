# hard clustering indices interface + general docs

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
    for i in eachindex(assignments)
        (assignments[i] in axes(centers, 2)) || throw(ArgumentError("Bad assignments[$i]=$(assignments[i]) is not a valid index for `data`."))
    end

    if quality_index == :calinski_harabasz
        _cluquality_calinski_harabasz(data, centers, assignments, metric)
    elseif quality_index == :xie_beni
        _cluquality_xie_beni(data, centers, assignments, metric)
    elseif quality_index == :davies_bouldin
        _cluquality_davies_bouldin(data, centers, assignments, metric)
    else quality_index == :davies_bouldin
    if quality_index == :silhouettes
        mean(silhouettes(assignments, pairwise(metric, eachcol(data))))
    elseif quality_index == :dunn 
        _cluquality_dunn(assignments, pairwise(metric, eachcol(data)))
    else
        throw(ArgumentError("Quality index $quality_index not supported."))
    end
end
end

clustering_quality(data::AbstractMatrix{<:Real}, R::KmeansResult; quality_index::Symbol, metric::SemiMetric=SqEuclidean()) =
    clustering_quality(data, R.centers, R.assignments; quality_index = quality_index, metric = metric)


# fuzzy clustering indices interface

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
        _cluquality_calinski_harabasz(data, centers, weights, fuzziness, metric)
    elseif quality_index ∈ (:xie_beni, :Xie_Beni, :xb)
        _cluquality_xie_beni(data, centers, weights, fuzziness, metric)
    else
        throw(ArgumentError("Quality index $quality_index not supported."))
    end
end

clustering_quality(data::AbstractMatrix{<:Real}, R::FuzzyCMeansResult; quality_index::Symbol, fuzziness::Real, metric::SemiMetric=SqEuclidean()) =
    clustering_quality(data, R.centers, R.weights; quality_index = quality_index, fuzziness = fuzziness, metric = metric)


# clustering indices with cluster centres not known interface

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
        error(ArgumentError("Quality index $quality_index not available."))
    end
end


clustering_quality(data::AbstractMatrix{<:Real}, assignments::AbstractVector{<:Integer}; quality_index::Symbol, metric::SemiMetric=SqEuclidean()) = 
    clustering_quality(assignments, pairwise(metric,eachcol(data)); quality_index = quality_index)

clustering_quality(data::AbstractMatrix{<:Real}, R::ClusteringResult;  quality_index::Symbol, metric::SemiMetric=SqEuclidean()) =
    clustering_quality(R.assignments, pairwise(metric,eachcol(data)); quality_index = quality_index)

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


function _inner_inertia(data, centers, cluster_samples, metric) # shared between hard clustering calinski_harabasz and xie_beni
    inner_inertia = sum(
        sum(colwise(metric, view(data, :, samples), center))
            for (center, samples) in zip(eachcol(centers), cluster_samples)
    )
    return inner_inertia
end

function _inner_inertia(data, centers, weights, fuzziness, metric) # shared between fuzzy clustering calinski_harabasz and xie_beni

    pointCentreDistances = pairwise(metric, eachcol(data), eachcol(centers))

    inner_inertia = sum(
        w^fuzziness * d for (w, d) in zip(weights, pointCentreDistances) 
    )

    return inner_inertia
end

# Calinski-Harabasz index

function  _cluquality_calinski_harabasz(
        data::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        metric::SemiMetric=SqEuclidean()
    )

    n, k = size(data, 2), size(centers, 2)

    cluster_samples = _gather_samples(assignments, k)
    global_center = vec(mean(data, dims=2))
    center_distances = colwise(metric, centers, global_center)
    outer_inertia = length.(cluster_samples) ⋅ center_distances

    inner_inertia = _inner_inertia(data, centers, cluster_samples, metric)

    return (outer_inertia / inner_inertia) * (n - k) / (k - 1)
end

function _cluquality_calinski_harabasz(
        data::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        weights::AbstractMatrix{<:Real},
        fuzziness::Real,
        metric::SemiMetric=SqEuclidean()
    )

    n, k = size(data, 2), size(centers, 2)

    global_center = vec(mean(data, dims=2))
    center_distances = colwise(metric, centers, global_center)

    outer_inertia = 
        sum(sum(w^fuzziness for w in w_col) * d
            for (w_col, d) in zip(eachcol(weights), center_distances)
        )
    inner_inertia = _inner_inertia(data, centers, weights, fuzziness, metric)

    return (outer_inertia / inner_inertia) * (n - k) / (k - 1)
end

# Davies-Bouldin index 

function _cluquality_davies_bouldin(
        data::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        metric::SemiMetric=SqEuclidean()
    )

    k = size(centers, 2)
    c_idx = axes(centers, 2)

    cluster_samples = _gather_samples(assignments, k)

    cluster_diameters = [mean(colwise(metric,view(data, :, sample), centers[:,j])) for (j, sample) in zip(c_idx, cluster_samples) ]
    center_distances = pairwise(metric,centers)

    DB = mean(
        maximum( (cluster_diameters[j₁] + cluster_diameters[j₂]) / center_distances[j₁,j₂] for j₂ in c_idx if j₂ ≠ j₁)
            for j₁ in c_idx
    )
    return  DB
end


# Xie-Beni index

function _cluquality_xie_beni(
        data::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        metric::SemiMetric=SqEuclidean()
    )

    n, k = size(data, 2), size(centers,2)

    cluster_samples = _gather_samples(assignments, k)
    inner_intertia  = _inner_inertia(data, centers, cluster_samples, metric)

    center_distances = pairwise(metric,centers)
    min_center_distance = minimum(center_distances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k)
    
    return inner_intertia / (n * min_center_distance)
end

function _cluquality_xie_beni(
        data::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        weights::AbstractMatrix{<:Real},
        fuzziness::Real,
        metric::SemiMetric=SqEuclidean()
    )

    n, k = size(data, 2), size(centers, 2)

    inner_intertia = _inner_inertia(data, centers, weights, fuzziness, metric)

    center_distances = pairwise(metric, eachcol(centers))
    min_center_distance = minimum(center_distances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k)

    return inner_intertia / (n * min_center_distance)
end


# Dunn index

function _cluquality_dunn(assignments::AbstractVector{<:Integer}, dist::AbstractMatrix{<:Real})

    max_inner_distance, min_outer_distance = typemin(eltype(dist)), typemax(eltype(dist))
    
    for i in eachindex(assignments), j in (i + 1):lastindex(assignments)
        d = dist[i,j]
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
