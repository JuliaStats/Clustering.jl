using Distances
# 1-based indexing

function calinski_harabasz(
        X::AbstractMatrix{<:Real},
        centers::AbstractMatrix{<:AbstractFloat},
        assignments::AbstractVector{<:Integer},
        distance::SemiMetric=SqEuclidean()
    )

    _check_qualityindex_argument(X,centers,assignments)

    n, k = size(X, 2), size(centers,2)

    innerInertia = sum(
        sum(colwise(distance, view(X, :, assignments .== j), centers[:, j])) for j in 1:k
    )

    counts = [count(==(j)assignment) for j in 1:k]
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
