


function clustering_quality(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean();
        quality_index::Symbol
    )
    d, n = size(X)
    _, data_idx = axes(X)
    dc, k = size(centers)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (1 <= k <= n) || throw(ArgumentError("Number of clusters k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    for i in eachindex(assignments)
        (assignments[i] in data_idx) || throw(ArgumentError("Bad assignments[$i]=$(assignments[i]) is not a valid index for `X`."))
    end


    if quality_index ∈ (:silhouettes, :silhouette, :s)
    elseif quality_index ∈ (:calinski_harabasz, :Calinski-Harabasz, :ch)
    elseif quality_index ∈ (:xie_beni, :Xie-Beni, :xb)
    elseif quality_index ∈ (:davies_bouldin, :Davies-Bouldin, :db)
    elseif quality_index ∈ (:dunn, Dunn, :d)
    else
        error(ArgumentError("Quality index $quality_index not available."))
    end
end

clustering_quality(X::AbstractMatrix{<:Real}, R::KmeansResult, distance::SemiMetric=SqEuclidean(); quality_index::Symbol) =
    clustering_quality(X, R.centers, R.assignments, distance; quality_index = quality_index)


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


end

clustering_quality(X::AbstractMatrix{<:Real}, R::FuzzyCMeansResult, fuzziness::Real, distance::SemiMetric=SqEuclidean(); quality_index::Symbol) =
    clustering_quality(X, R.centers, R.weights, fuzziness, distance; quality_index)

function clustering_quality(
        assignments::AbstractVector{<:Integer},
        dist::AbstractMatrix{<:Real};
        quality_index::Symbol = :dunn
    )
    n, m = size(dist)
    na = length(assignments)
    n == m || throw(ArgumentError("Distance matrix must be square."))
    n == na || throw(DimensionMismatch("Inconsistent array dimensions for distance matrix and assignments."))

end


clustering_quality(X::AbstractMatrix{<:Real}, assignments::AbstractVector{<:Integer}, distance::SemiMetric=SqEuclidean(); quality_index::Symbol = :dunn) = 
    clustering_quality(assignments, pairwise(distance,eachcol(X)); quality_index)

clustering_quality(X::AbstractMatrix{<:Real}, R::ClusteringResult, distance::SemiMetric=SqEuclidean(); quality_index::Symbol = :dunn) =
    clustering_quality(X, R.assignments, distance; quality_index)

clustering_quality(R::ClusteringResult, dist::AbstractMatrix{<:Real}; quality_index::Symbol = :dunn) = 
    clustering_quality(R.assignments, dist; quality_index)

function _check_qualityindex_arguments(
        X::AbstractMatrix{<:Real},              # data matrix (d x n)
        centers::AbstractMatrix{<:Real},        # cluster centers (d x k)
        assignments::AbstractVector{<:Integer}, # assignments (n)
    )
    d, n = size(X)
    _, data_idx = axes(X)
    dc, k = size(centers)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (1 <= k <= n) || throw(ArgumentError("Number of clusters k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    for i in eachindex(assignments)
        (assignments[i] in data_idx) || throw(ArgumentError("Bad assignments[$i]=$(assignments[i]) is not a valid index for `X`."))
    end
end

function _check_qualityindex_arguments(
        X::AbstractMatrix{<:Real},       # data matrix (d x n)
        centers::AbstractMatrix{<:Real}, # cluster centers (d x k)
        weights::AbstractMatrix{<:Real}, # assigned weights (n x k)
        fuzziness::Real,                 # cluster fuzziness
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
end

function _check_qualityindex_arguments(
        assignments::AbstractVector{<:Integer}, # assignments (n)
        dist::AbstractMatrix{<:Real}            # data distance matrix (n x n)
    )
    n, m = size(dist)
    na = length(assignments)
    n == m || throw(ArgumentError("Distance matrix must be square."))
    n == na || throw(DimensionMismatch("Inconsistent array dimensions for distance matrix and assignments."))
end


function _gather_samples(assignments, k)
    cluster_samples = [Int[] for _ in  1:k]
    for (i, a) in enumerate(assignments)
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

function calinski_harabasz(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean()
    )
    _check_qualityindex_arguments(X, centers, assignments)

    n, k = size(X, 2), size(centers, 2)

    cluster_samples = _gather_samples(assignments, k)
    global_center = vec(mean(X, dims=2))
    center_distances = colwise(distance, centers, global_center)
    outer_inertia = length.(cluster_samples) ⋅ center_distances

    inner_inertia = _inner_inertia(X, centers, cluster_samples, distance)

    return (outer_inertia / inner_inertia) * (n - k) / (k - 1)
end

calinski_harabasz(X::AbstractMatrix{<:Real}, R::KmeansResult, distance::SemiMetric=SqEuclidean()) =
calinski_harabasz(X, R.centers, R.assignments, distance)


function calinski_harabasz(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        weights::AbstractMatrix{<:Real},
        fuzziness::Real,
        distance::SemiMetric=SqEuclidean()
    )
    _check_qualityindex_arguments(X, centers, weights, fuzziness)

    n, k = size(X, 2), size(centers,2)
    w_idx1, w_idx2 = axes(weights)

    global_center = vec(mean(X, dims=2))
    center_distances = colwise(distance, centers, global_center)
    outer_intertia = sum(
        weights[i,j₁]^fuzziness * center_distances[j₂] for i in w_idx1, (j₁,j₂) in zip(w_idx2, 1:k)
    )

    inner_intertia = _inner_inertia(X, centers, weights, fuzziness, distance)

    return (outer_intertia / (k - 1)) / (inner_intertia / (n - k))
end

calinski_harabasz(X::AbstractMatrix{<:Real}, R::FuzzyCMeansResult, fuzziness::Real, distance::SemiMetric=SqEuclidean()) =
    calinski_harabasz(X, R.centers, R.weights, fuzziness, distance)


# Davies-Bouldin idex 

function davies_bouldin(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean()
    )
    _check_qualityindex_arguments(X, centers, assignments)

    k = size(centers, 2)
    c_idx = axes(centers,2)

    cluster_samples = _gather_samples(assignments, k)

    cluster_diameters = [mean(colwise(distance,view(X, :, sample), centers[:,j])) for (j, sample) in zip(c_idx, cluster_samples) ]
    center_distances = pairwise(distance,centers)

    DB = mean(
        maximum( (cluster_diameters[j₁] + cluster_diameters[j₂]) / center_distances[j₁,j₂] for j₂ in c_idx if j₂ ≠ j₁)
            for j₁ in c_idx
    )
    return  DB
end

davies_bouldin(X::AbstractMatrix{<:Real}, R::KmeansResult, distance::SemiMetric=SqEuclidean()) =
    davies_bouldin(X, R.centers, R.assignments, distance)


# Xie-Beni index

function xie_beni(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean()
    )
    _check_qualityindex_arguments(X, centers, assignments)

    n, k = size(X, 2), size(centers,2)

    cluster_samples = _gather_samples(assignments, k)
    inner_intertia  = _inner_inertia(X, centers, cluster_samples, distance)

    center_distances = pairwise(distance,centers)
    min_center_distance = minimum(center_distances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k)
    
    return inner_intertia / (n * min_center_distance)
end

xie_beni(X::AbstractMatrix{<:Real}, R::KmeansResult, distance::SemiMetric=SqEuclidean()) =
    xie_beni(X, R.centers, R.assignments, distance)


function xie_beni(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:Real},
        weights::AbstractMatrix{<:Real},
        fuzziness::Real,
        distance::SemiMetric=SqEuclidean()
    )
    _check_qualityindex_arguments(X, centers, weights, fuzziness)

    n, k = size(X, 2), size(centers, 2)

    inner_intertia = _inner_inertia(X, centers, weights, fuzziness, distance)

    center_distances = pairwise(distance, eachcol(centers))
    min_center_distance = minimum(center_distances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k)

    return inner_intertia / (n * min_center_distance)
end

xie_beni(X::AbstractMatrix{<:Real}, R::FuzzyCMeansResult, fuzziness::Real, distance::SemiMetric=SqEuclidean()) =
    xie_beni(X, R.centers, R.weights, fuzziness, distance)


# Dunn index

function dunn(assignments::AbstractVector{<:Integer}, dist::AbstractMatrix{<:Real})
    _check_qualityindex_arguments(assignments, dist)

    k = maximum(assignments)
    cluster_samples = _gather_samples(assignments, k)

    min_outer_distance = minimum(
        minimum(view(dist, cluster_samples[j₁], cluster_samples[j₂]))
            for j₁ in 1:k for j₂ in j₁+1:k
    )

    max_inner_distance = maximum(
        maximum(dist[i₁,i₂] for i₁ in sample, i₂ in sample)
            for sample in cluster_samples
    )
    
    return min_outer_distance / max_inner_distance
end

dunn(X::AbstractMatrix{<:Real}, assignments::AbstractVector{<:Integer}, distance::SemiMetric=SqEuclidean()) = 
    dunn(assignments, pairwise(distance,eachcol(X)))

dunn(X::AbstractMatrix{<:Real}, R::ClusteringResult, distance::SemiMetric=SqEuclidean()) =
    dunn(X, R.assignments, distance)

dunn(R::ClusteringResult, dist::AbstractMatrix{<:Real}) = dunn(R.assignments, dist)