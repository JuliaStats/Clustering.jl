using Distances
# 1-based indexing

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
