# Common utilities

##### common types

"""
    ClusteringResult

Base type for the output of clustering algorithm.
"""
abstract type ClusteringResult end

# vector of cluster indices for each clustered point
ClusterAssignments = AbstractVector{<:Integer}

ClusteringResultOrAssignments = Union{ClusteringResult, ClusterAssignments}

# generic functions

"""
    nclusters(R::ClusteringResult) -> Int

Get the number of clusters.
"""
nclusters(R::ClusteringResult) = length(R.counts)

"""
    counts(R::ClusteringResult) -> Vector{Int}

Get the vector of cluster sizes.

`counts(R)[k]` is the number of points assigned to the ``k``-th cluster.
"""
counts(R::ClusteringResult) = R.counts

"""
    wcounts(R::ClusteringResult) -> Vector{Float64}
    wcounts(R::FuzzyCMeansResult) -> Vector{Float64}

Get the weighted cluster sizes as the sum of weights of points assigned to each
cluster.

For non-weighted clusterings assumes the weight of every data point is 1.0,
so the result is equivalent to `convert(Vector{Float64}, counts(R))`.
"""
wcounts(R::ClusteringResult) = convert(Vector{Float64}, counts(R))

"""
    assignments(R::ClusteringResult) -> Vector{Int}

Get the vector of cluster indices for each point.

`assignments(R)[i]` is the index of the cluster to which the ``i``-th point
is assigned.
"""
assignments(R::ClusteringResult) = R.assignments
assignments(A::ClusterAssignments) = A


##### convert display symbol to disp level
const DisplayLevels = Dict(:none => 0, :final => 1, :iter => 2)

display_level(s::Symbol) = get(DisplayLevels, s) do
    valid_vals = string.(":", first.(sort!(collect(pairs(DisplayLevels)), by=last)))
    throw(ArgumentError("Invalid option display=:$s ($(join(valid_vals, ", ", ", or ")) expected)"))
end

function check_assignments(assignments::AbstractVector{<:Integer}, nclusters::Union{Integer, Nothing})
    nclu = nclusters === nothing ? maximum(assignments) : nclusters
    for (j, c) in enumerate(assignments)
        all(1 <= c <= nclu) || throw(ArgumentError("Bad assignments[$j]=$c: should be in 1:$nclu range."))
    end
end