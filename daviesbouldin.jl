

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