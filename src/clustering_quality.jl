

"""
    clustering_quality(X, centers, assignments, [distance;] quality_index)
    clustering_quality(X, kmeans_clustering, [distance;] quality_index)

Compute the clustering quality index for a given clustering.

# Arguments
 - `data::AbstractMatrix`: ``d×n`` data matrix with each column representing one ``d``-dimensional data point
 - `centers::AbstractMatrix`: ``d×k`` matrix with cluster centers represented as columns
 - `assignments::AbstractVector{Int}`: ``n`` vector of point assignments (cluster indices)
 - `clustering::ClusteringResult`: the output of the clustering method
 - `distance::SemiMetric=SqEuclidean()`: `SemiMetric` object that defines the distance between the data points
 - `quality_index::Symbol`: quality index to calculate; see below for the supported options

# Supported quality indices

Please refer to the [documentation](@ref clustering_quality) for the extended description of the quality indices.

- `:silhouettes`: average silhouette index, for all silhouettes use [`silhouettes`](@ref) method instead
- `:calinski_harabasz`: Calinski-Harabsz index, the corrected ratio of inertia between cluster centers and within-clusters inertia
- `:xie_beni`: Xie-Beni index (↓) returns ratio betwen inertia within clusters and minimal distance between cluster centers
- `:davies_bouldin`: Davies-Bouldin index (↓) returns average similarity between each cluster and its most similar one, averaged over all the clusters
- `:silhouettes`: average silhouette index (↑), to obtain all silhouettes use `silhouettes` function instead, it does not make use of `centers` argument
- `:dunn`: Dunn index (↑) returns ratio between minimal distance between clusters and maximal cluster diameter, it does not make use of `centers` argument

"""
function clustering_quality(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean();
        quality_index::Symbol
    )
    d, n = size(X)
    dc, k = size(centers)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (1 <= k <= n) || throw(ArgumentError("Number of clusters k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    for i in eachindex(assignments)
        (assignments[i] in axes(centers, 2)) || throw(ArgumentError("Bad assignments[$i]=$(assignments[i]) is not a valid index for `X`."))
    end

    if quality_index == :calinski_harabasz
        _cluquality_calinski_harabasz(X, centers, assignments, distance)
    elseif quality_index ∈ (:xie_beni, :Xie_Beni, :xb)
        _cluquality_xie_beni(X, centers, assignments, distance)
    elseif quality_index ∈ (:davies_bouldin, :Davies_Bouldin, :db)
        _cluquality_davies_bouldin(X, centers, assignments, distance)
    else quality_index ∈ (:davies_bouldin, :Davies_Bouldin, :db)
    if quality_index ∈ (:silhouettes, :silhouette, :s)
        mean(silhouettes(assignments, pairwise(distance, eachcol(X))))
    elseif quality_index ∈ (:dunn, :Dunn, :d)
        _cluquality_dunn(assignments, pairwise(distance, eachcol(X)))
    else
        throw(ArgumentError("Quality index $quality_index not supported."))
    end
end
end

clustering_quality(X::AbstractMatrix{<:Real}, R::KmeansResult, distance::SemiMetric=SqEuclidean(); quality_index::Symbol) =
    clustering_quality(X, R.centers, R.assignments, distance; quality_index = quality_index)

"""
    clustering_quality(X, centers, weights, fuzziness, [distance;] quality_index)
    clustering_quality(X, fuzzy_cmeans_clustering, fuzziness, [distance;] quality_index)

Compute chosen quality index  value for a soft (fuzzy) clustering 

# Arguments
 - `data::AbstractMatrix`: ``d×n`` data matrix with each column representing one ``d``-dimensional data point
 - `centers::AbstractMatrix`: ``d×k`` matrix with cluster centers represented as columns
 - `weights::AbstractMatrix`: ``n×k`` matrix with fuzzy clustering weights, `weights[i,j]` is the degree of membership of ``i``-th data point to ``j``-th cluster
 - `fuzziness::Real`: clustering fuzziness > 1
 - `fuzzy_cmeans_clustering::FuzzyCMeansResult`: the output of fuzzy_cmeans method
 - `distance::SemiMetric=SqEuclidean()`: `SemiMetric` object that defines the distance between the data points
 - `quality_index::Symbol`: chosen quality index

 # Available quality indices:
 Depending on the index higher (↑) or lower (↓) value suggests better clustering quality.
 
 - `:calinski_harabasz`: Calinski-Harabsz index (↑) returns corrected ratio between inertia between cluster centers and inertia within clusters
 - `:xie_beni`: Xie-Beni index (↓) returns ratio betwen inertia within clusters and minimal distance between cluster centers

"""
function clustering_quality(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        weights::AbstractMatrix{<:Real},
        fuzziness::Real,
        distance::SemiMetric=SqEuclidean();
        quality_index::Symbol
    )
    d, n = size(X)
    dc, k = size(centers)
    nw, kw = size(weights)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    n == nw || throw(DimensionMismatch("Inconsistent data length for `X` and `weights`."))
    k == kw || throw(DimensionMismatch("Inconsistent number of clusters for `centers` and `weights`."))
    (1 <= k <= n) || throw(ArgumentError("Number of clusters k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    all(>=(0), weights) || throw(ArgumentError("All weights must be larger or equal 0."))
    1 < fuzziness || throw(ArgumentError("Fuzziness must be greater than 1 ($fuzziness given)"))

    if quality_index == :calinski_harabasz
        _cluquality_calinski_harabasz(X, centers, weights, fuzziness, distance)
    elseif quality_index ∈ (:xie_beni, :Xie_Beni, :xb)
        _cluquality_xie_beni(X, centers, weights, fuzziness, distance)
    else
        throw(ArgumentError("Quality index $quality_index not supported."))
    end
end

clustering_quality(X::AbstractMatrix{<:Real}, R::FuzzyCMeansResult, fuzziness::Real, distance::SemiMetric=SqEuclidean(); quality_index::Symbol) =
    clustering_quality(X, R.centers, R.weights, fuzziness, distance; quality_index)

"""

    clustering_quality(assignments, dist_matrix; quality_index)
    clustering_quality(clustering, dist_matrix; quality_index)
    clustering_quality(data, assignments, [distance;] quality_index)
    clustering_quality(data, clustering, [distance;] quality_index)

Compute chosen quality index value for a clustering in a case cluster centres may be not known. 

# Arguments
 - `data::AbstractMatrix`: ``d×n`` data matrix with each column representing one ``d``-dimensional data point
 - `assignments::AbstractVector{Int}`: the vector of point assignments (cluster indices)
 - `dist_matrix::AbstractMatrix`: a ``n×n`` pairwise distance matrix; `dist_matrix[i,j]` is the distance between ``i``-th and ``j``-th points.
 - `distance::SemiMetric=SqEuclidean()`:  `SemiMetric` object that defines the distance between the data points
 - `clustering::ClusteringResult`: the output of some clustering method
 - `quality_index::Symbol`: chosen quality index

# Available quality indices:
Depending on the index higher (↑) or lower (↓) value suggests better clustering quality.

- `:silhouettes`: average silhouette index (↑), to obtain all silhouettes use `silhouettes` function instead
- `:dunn`: Dunn index (↑) returns ratio between minimal distance between clusters and maximal cluster diameter

"""
function clustering_quality(
        assignments::AbstractVector{<:Integer},
        dist::AbstractMatrix{<:Real};
        quality_index::Symbol 
    )
    n, m = size(dist)
    na = length(assignments)
    n == m || throw(ArgumentError("Distance matrix must be square."))
    n == na || throw(DimensionMismatch("Inconsistent array dimensions for distance matrix and assignments."))

    if quality_index == :silhouettes
        mean(silhouettes(assignments, dist))
    elseif quality_index ∈ (:dunn, :Dunn, :d)
        _cluquality_dunn(assignments, dist)
    else
        error(ArgumentError("Quality index $quality_index not available."))
    end
end


clustering_quality(X::AbstractMatrix{<:Real}, assignments::AbstractVector{<:Integer}, distance::SemiMetric=SqEuclidean(); quality_index::Symbol) = 
    clustering_quality(assignments, pairwise(distance,eachcol(X)); quality_index = quality_index)

clustering_quality(X::AbstractMatrix{<:Real}, R::ClusteringResult, distance::SemiMetric=SqEuclidean(); quality_index::Symbol) =
    clustering_quality(R.assignments, pairwise(distance,eachcol(X)); quality_index = quality_index)

clustering_quality(R::ClusteringResult, dist::AbstractMatrix{<:Real}; quality_index::Symbol) = 
    clustering_quality(R.assignments, dist; quality_index = quality_index)


function _gather_samples(assignments, k) # cluster_samples[j]: indices of points in cluster j
    cluster_samples = [Int[] for _ in  1:k]
    for (i, a) in zip(eachindex(assignments), assignments)
        push!(cluster_samples[a], i)
    end
    return cluster_samples
end


function _inner_inertia(X, centers, cluster_samples, distance) # shared between hard clustering calinski_harabasz and xie_beni
    inner_inertia = sum(
        sum(colwise(distance, view(X, :, samples), center))
            for (center, samples) in zip(eachcol(centers), cluster_samples)
    )
    return inner_inertia
end

function _inner_inertia(X, centers, weights, fuzziness, distance) # shared between soft clustering calinski_harabasz and xie_beni
    n, k = size(X, 2), size(centers, 2)
    w_idx1, w_idx2 = axes(weights)
    pointCentreDistances = pairwise(distance, eachcol(X), eachcol(centers))
    inner_inertia = sum(
        weights[i₁,j₁]^fuzziness * pointCentreDistances[i₂,j₂] for (i₁,i₂) in zip(w_idx1,1:n), (j₁,j₂) in zip(w_idx2, 1:k)
    )
    return inner_inertia
end

# Calinski-Harabasz index

function  _cluquality_calinski_harabasz(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean()
    )

    n, k = size(X, 2), size(centers, 2)

    cluster_samples = _gather_samples(assignments, k)
    global_center = vec(mean(X, dims=2))
    center_distances = colwise(distance, centers, global_center)
    outer_inertia = length.(cluster_samples) ⋅ center_distances

    inner_inertia = _inner_inertia(X, centers, cluster_samples, distance)

    return (outer_inertia / inner_inertia) * (n - k) / (k - 1)
end

function _cluquality_calinski_harabasz(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        weights::AbstractMatrix{<:Real},
        fuzziness::Real,
        distance::SemiMetric=SqEuclidean()
    )

    n, k = size(X, 2), size(centers, 2)
    w_idx1, w_idx2 = axes(weights)

    global_center = vec(mean(X, dims=2))
    center_distances = colwise(distance, centers, global_center)
    outer_intertia = sum(
        weights[i,j₁]^fuzziness * center_distances[j₂] for i in w_idx1, (j₁,j₂) in zip(w_idx2, 1:k)
    )

    inner_intertia = _inner_inertia(X, centers, weights, fuzziness, distance)

    return (outer_intertia / inner_inertia) * (n - k) / (k - 1)
end

# Davies-Bouldin idex 

function _cluquality_davies_bouldin(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean()
    )

    k = size(centers, 2)
    c_idx = axes(centers, 2)

    cluster_samples = _gather_samples(assignments, k)

    cluster_diameters = [mean(colwise(distance,view(X, :, sample), centers[:,j])) for (j, sample) in zip(c_idx, cluster_samples) ]
    center_distances = pairwise(distance,centers)

    DB = mean(
        maximum( (cluster_diameters[j₁] + cluster_diameters[j₂]) / center_distances[j₁,j₂] for j₂ in c_idx if j₂ ≠ j₁)
            for j₁ in c_idx
    )
    return  DB
end


# Xie-Beni index

function _cluquality_xie_beni(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean()
    )

    n, k = size(X, 2), size(centers,2)

    cluster_samples = _gather_samples(assignments, k)
    inner_intertia  = _inner_inertia(X, centers, cluster_samples, distance)

    center_distances = pairwise(distance,centers)
    min_center_distance = minimum(center_distances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k)
    
    return inner_intertia / (n * min_center_distance)
end

function _cluquality_xie_beni(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        weights::AbstractMatrix{<:Real},
        fuzziness::Real,
        distance::SemiMetric=SqEuclidean()
    )

    n, k = size(X, 2), size(centers, 2)

    inner_intertia = _inner_inertia(X, centers, weights, fuzziness, distance)

    center_distances = pairwise(distance, eachcol(centers))
    min_center_distance = minimum(center_distances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k)

    return inner_intertia / (n * min_center_distance)
end


# Dunn index

function _cluquality_dunn(assignments::AbstractVector{<:Integer}, dist::AbstractMatrix{<:Real})

    k = maximum(assignments)

    cluster_samples = _gather_samples(assignments, k)

    min_outer_distance = minimum(
        minimum(view(dist, cluster_samples[j₁], cluster_samples[j₂]), init = typemax(eltype(dist)))
            for j₁ in 1:k for j₂ in j₁+1:k
    )

    max_inner_distance = maximum(
        maximum(dist[i₁,i₂] for i₁ in sample, i₂ in sample, init = typemin(eltype(dist)))
            for sample in cluster_samples
    )
    
    return min_outer_distance / max_inner_distance
end
