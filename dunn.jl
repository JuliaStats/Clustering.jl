



function dunn(assignments::AbstractVector{<:Integer}, dist::AbstractMatrix{<:Real})
    _check_qualityindex_argument(assignments, dist)

    n = size(dist, 1)
    k = maximum(assignments)
end


dunn(X::AbstractMatrix{<:Real}, assignments::AbstractVector{<:Integer}, distance::SemiMetric=SqEuclidean()) = 
    dunn(assignments, pairwise(distance,X))

dunn(X::AbstractMatrix{<:Real}, R::ClusteringResult, distance::SemiMetric=SqEuclidean()) =
    dunn(X, R.assignments, distance)

dunn(R::ClusteringResult, dist::AbstractMatrix{<:Real}) = dunn(R.assignments, dist)