using Distances
# 1-based indexing

# argument checking

function _check_qualityindex_argument(
        X::AbstractMatrix{<:Real},                  # data matrix (d x n)
        centers::AbstractMatrix{<:AbstractFloat},   # cluster centers (d x k)
        assignments::AbstractVector{<:Integer},     # assignments (n)
    )
    d, n = size(X)
    dc, k = size(centers)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (1 <= k <= n) || throw(ArgumentError("Cluster number k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    for j = 1:n
        (1 <= assignments[j] <= k) || throw(ArgumentError("Bad assignments[$j]=$(assignments[j]): should be in 1:$k range."))
    end
end

function _check_qualityindex_argument(
        X::AbstractMatrix{<:Real},                # data matrix (d x n)
        centers::AbstractMatrix{<:AbstractFloat}, # cluster centers (d x k)
        weights::AbstractMatrix{<:AbstractFloat}, # assigned weights (n x C)
        fuzziness::Real,                          # cluster fuzziness 
    )
    d, n = size(X)
    dc, k = size(centers)
    nw, kw = size(weights)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    n == nw || throw(DimensionMismatch("Inconsistent data length for `X` and `weights`."))
    k == kw || throw(DimensionMismatch("Inconsistent number of clusters for `centers` and `weights`."))
    (1 <= k <= n) || throw(ArgumentError("Cluster number k must be from 1:n (n=$n), k=$k given."))
    k >= 2 || throw(ArgumentError("Quality index not defined for the degenerated clustering with a single cluster."))
    n == k && throw(ArgumentError("Quality index not defined for the degenerated clustering where each data point is its own cluster."))
    all(>=(0), weights) || throw(ArgumentError("All weights must be larger or equal 0."))
    1 < fuzziness || throw(ArgumentError("Fuzziness must be greater than 1 ($fuzziness given)"))
end

function _check_qualityindex_argument(
        assignments::AbstractVector{<:Integer}, # assignments (n)
        dist::AbstractMatrix{<:Real}            # data distance matrix (n x n)
    )
    n, m = size(dist)
    na = length(assignments)
    n == m || throw(ArgumentError("Distance matrix must be square."))
    n == na || throw(DimensionMismatch("Inconsistent array dimensions for distance matrix and assignments."))
end

# Calinski-Harabasz index

function calinski_harabasz(
    X::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:AbstractFloat},
    assignments::AbstractVector{<:Integer},
    distance::SemiMetric=SqEuclidean()
)
_check_qualityindex_argument(X, centers, assignments)

n, k = size(X, 2), size(centers,2)

innerInertia = sum(
    sum(colwise(distance, view(X, :, assignments .== j), centers[:, j])) for j in 1:k
)

counts = [count(==(j), assignments) for j in 1:k]
globalCenter = mean(X, dims=2)
outerInertia = sum(
    counts[j] * distance(centers[:, j], globalCenter) for j in 1:k
)

return (outerInertia / (k - 1)) / (innerInertia / (n - k))
end

calinski_harabasz(X::AbstractMatrix{<:Real}, R::KmeansResult, distance::SemiMetric=SqEuclidean()) =
calinski_harabasz(X, R.centers, R.assignments, distance)


function calinski_harabasz(
    X::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:AbstractFloat},
    weights::AbstractMatrix{<:AbstractFloat},
    fuzziness::Real,
    distance::SemiMetric=SqEuclidean()
)
_check_qualityindex_argument(X, centers, weights, fuzziness)

n, k = size(X, 2), size(centers,2)

innerInertia = sum(
    weights[i,j]^fuzziness * distance(X[:,i],centers[:,j]) for i in 1:n, j in 1:k
)

globalCenter = mean(X, dims=2)[:]
centerDistances = colwise(distance, centers, globalCenter)
outerInertia = sum(
    weights[i,j]^fuzziness * centerDistances[j] for i in 1:n, j in 1:k
)

return (outerInertia / (k - 1)) / (innerInertia / (n - k))
end

calinski_harabasz(X::AbstractMatrix{<:Real}, R::FuzzyCMeansResult, fuzziness::Real, distance::SemiMetric=SqEuclidean()) =
calinski_harabasz(X, R.centers, R.weights, fuzziness, distance)


# Davies-Bouldin idex 

function davies_bouldin(
    X::AbstractMatrix{<:Real},
    centers::AbstractMatrix{<:AbstractFloat},
    assignments::AbstractVector{<:Integer},
    distance::SemiMetric=SqEuclidean()
)
_check_qualityindex_argument(X, centers, assignments)

n, k = size(X, 2), size(centers,2)

centerDiameters = [mean(colwise(distance,view(X, :, assignments .== j), centers[:,j])) for j in 1:k ]
centerDistances = pairwise(distance,centers)

return  maximum( (centerDiameters[j₁] + centerDiameters[j₂]) / centerDistances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k ) / k
end

davies_bouldin(X::AbstractMatrix{<:Real}, R::KmeansResult, distance::SemiMetric=SqEuclidean()) =
davies_bouldin(X, R.centers, R.assignments, distance)


# Xie-Beni index

function xie_beni(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:AbstractFloat},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean()
    )
    _check_qualityindex_argument(X, centers, assignments)

    n, k = size(X, 2), size(centers,2)

    innerInertia = sum(
        sum(colwise(distance, view(X, :, assignments .== j), centers[:, j])) for j in 1:k
    )

    centerDistances = pairwise(distance,centers)
    minOuterDistance = minimum(centerDistances[j₁,j₂] for j₁ in 1:k for j₂ in j₁+1:k)
    return innerInertia / (n * minOuterDistance)
end

xie_beni(X::AbstractMatrix{<:Real}, R::KmeansResult, distance::SemiMetric=SqEuclidean()) =
    xie_beni(X, R.centers, R.assignments, distance)


function xie_beni(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:AbstractFloat},
        weights::AbstractMatrix{<:AbstractFloat},
        fuzziness::Real,
        distance::SemiMetric=SqEuclidean()
    )
    _check_qualityindex_argument(X, centers, weights, fuzziness)

    n, k = size(X, 2), size(centers,2)

    innerInertia = sum(
        weights[i,j]^fuzziness * distance(X[:,i],centers[:,j]) for i in 1:n, j in 1:k
    )

    centerDistances = pairwise(distance,centers)
    minOuterDistance = minimum(centerDistances[i,j] for i in 1:k for j in i+1:k)
    return innerInertia / (n * minOuterDistance)
end

xie_beni(X::AbstractMatrix{<:Real}, R::FuzzyCMeansResult, fuzziness::Real, distance::SemiMetric=SqEuclidean()) =
    xie_beni(X, R.centers, R.weights, fuzziness, distance)


# Dunn index

function dunn(assignments::AbstractVector{<:Integer}, dist::AbstractMatrix{<:Real})
    _check_qualityindex_argument(assignments, dist)

    n = size(dist, 1)
    k = maximum(assignments)

    minOuterDistance = eltype(dist)(Inf)
    
    for j₁ in 1:k, j₂ in j₁+1:k
        # δ is min distance between points from clusters j₁ and j₂
        δ = minimum(dist[i₁,i₂] for i₁ in findall(==(j₁), assignments), i₂ in findall(==(j₂), assignments))

        if δ < minOuterDistance
            minOuterDistance = δ
        end
    end

    maxInnerDistance = eltype(dist)(-Inf)

    for j in 1:k
        # Δ is max distance between points in cluster j
        Δ = maximum(dist[i₁,i₂] for i₁ in findall(==(j), assignments), i₂ in findall(==(j), assignments))

        if Δ > maxInnerDistance
            maxInnerDistance = Δ
        end
    end
    
    return minOuterDistance / maxInnerDistance
end

dunn(X::AbstractMatrix{<:Real}, assignments::AbstractVector{<:Integer}, distance::SemiMetric=SqEuclidean()) = 
    dunn(assignments, pairwise(distance,X))

dunn(X::AbstractMatrix{<:Real}, R::ClusteringResult, distance::SemiMetric=SqEuclidean()) =
    dunn(X, R.assignments, distance)

dunn(R::ClusteringResult, dist::AbstractMatrix{<:Real}) = dunn(R.assignments, dist)