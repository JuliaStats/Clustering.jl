# Initialization algorithms
#
#   Each algorithm is represented by a subtype of SeedingAlgorithm
#
#   Let alg be an instance of such an algorithm, then it should
#   support the following usage:
#
#       initseeds!(iseeds, alg, X)
#       initseeds_by_costs!(iseeds, alg, costs)
#
#   Here:
#       - iseeds:   a vector of resultant indexes of the chosen seeds
#       - alg:      the seeding algorithm instance
#       - X:        the data matrix, each column being a data point
#       - costs:    pre-computed pairwise cost matrix.
#
#   This function returns iseeds
#

"""
    SeedingAlgorithm

Base type for all seeding algorithms.

Each seeding algorithm should implement the two functions: [`initseeds!`](@ref)
and [`initseeds_by_costs!`](@ref).
"""
abstract type SeedingAlgorithm end

"""
    initseeds(alg::Union{SeedingAlgorithm, Symbol},
              X::AbstractMatrix, k::Integer) -> Vector{Int}

Select `k` seeds from a ``d×n`` data matrix `X` using the `alg`
algorithm.

`alg` could be either an instance of [`SeedingAlgorithm`](@ref) or a symbolic
name of the algorithm.

Returns the vector of `k` seed indices.
"""
initseeds(alg::SeedingAlgorithm, X::AbstractMatrix{<:Real}, k::Integer) =
    initseeds!(Vector{Int}(undef, k), alg, X)

"""
    initseeds_by_costs(alg::Union{SeedingAlgorithm, Symbol},
                       costs::AbstractMatrix, k::Integer) -> Vector{Int}

Select `k` seeds from the ``n×n`` `costs` matrix using algorithm `alg`.

Here, `costs[i, j]` is the cost of assigning points `i`` and ``j``
to the same cluster. One may, for example, use the squared Euclidean distance
between the points as the cost.

Returns the vector of `k` seed indices.
"""
initseeds_by_costs(alg::SeedingAlgorithm, costs::AbstractMatrix{<:Real}, k::Integer) =
    initseeds_by_costs!(Vector{Int}(undef, k), alg, costs)

seeding_algorithm(s::Symbol) =
    s == :rand ? RandSeedAlg() :
    s == :kmpp ? KmppAlg() :
    s == :kmcen ? KmCentralityAlg() :
    throw(ArgumentError("Unknown seeding algorithm $s"))

function check_seeding_args(n::Integer, k::Integer)
    k >= 1 || throw(ArgumentError("The number of seeds ($k) must be positive."))
    k <= n || throw(ArgumentError("Cannot select more seeds ($k) than data points ($n)."))
end

check_seeding_args(X::AbstractMatrix, iseeds::AbstractVector) =
    check_seeding_args(size(X, 2), length(iseeds))

initseeds(algname::Symbol, X::AbstractMatrix{<:Real}, k::Integer) =
    initseeds(seeding_algorithm(algname), X, k)::Vector{Int}

initseeds_by_costs(algname::Symbol, costs::AbstractMatrix{<:Real}, k::Integer) =
    initseeds_by_costs(seeding_algorithm(algname), costs, k)

# use specified vector of seeds
function initseeds(iseeds::AbstractVector{<:Integer}, X::AbstractMatrix{<:Real}, k::Integer)
    length(iseeds) == k ||
        throw(ArgumentError("The length of seeds vector ($(length(iseeds))) differs from the number of seeds requested ($k)"))
    check_seeding_args(X, iseeds)
    n = size(X, 2)
    # check that seed indices are fine
    for (i, seed) in enumerate(iseeds)
        (1 <= seed <= n) || throw(ArgumentError("Seed #$i refers to an incorrect data point ($seed)"))
    end
    # NOTE no duplicate checks are done, should we?
    convert(Vector{Int}, iseeds)
end
initseeds_by_costs(iseeds::AbstractVector{<:Integer}, costs::AbstractMatrix{<:Real}, k::Integer) =
    initseeds(iseeds, costs, k) # NOTE: passing costs as X, but should be fine since only size(X, 2) is used

function copyseeds!(S::AbstractMatrix{<:AbstractFloat},
                    X::AbstractMatrix{<:Real},
                    iseeds::AbstractVector)
    d, n = size(X)
    k = length(iseeds)
    size(S) == (d, k) ||
        throw(DimensionMismatch("Inconsistent seeds matrix dimensions: $((d, k)) expected, $(size(S)) given."))
    return copyto!(S, view(X, :, iseeds))
end

"""
    RandSeedAlg <: SeedingAlgorithm

Random seeding (`:rand`).

Chooses an arbitrary subset of ``k`` data points as cluster seeds.
"""
struct RandSeedAlg <: SeedingAlgorithm end

"""
    initseeds!(iseeds::AbstractVector{Int}, alg::SeedingAlgorithm,
               X::AbstractMatrix) -> iseeds

Initialize `iseeds` with the indices of cluster seeds for the `X` data matrix
using the `alg` seeding algorithm.
"""
function initseeds!(iseeds::IntegerVector, alg::RandSeedAlg, X::AbstractMatrix{<:Real})
    check_seeding_args(X, iseeds)
    sample!(1:size(X, 2), iseeds; replace=false)
end

"""
    initseeds_by_costs!(iseeds::AbstractVector{Int}, alg::SeedingAlgorithm,
                        costs::AbstractMatrix) -> iseeds

Initialize `iseeds` with the indices of cluster seeds for the `costs` matrix
using the `alg` seeding algorithm.

Here, `costs[i, j]` is the cost of assigning points ``i`` and ``j``
to the same cluster. One may, for example, use the squared Euclidean distance
between the points as the cost.
"""
function initseeds_by_costs!(iseeds::IntegerVector, alg::RandSeedAlg, X::AbstractMatrix{<:Real})
    check_seeding_args(X, iseeds)
    sample!(1:size(X,2), iseeds; replace=false)
end

"""
    KmppAlg <: SeedingAlgorithm

Kmeans++ seeding (`:kmpp`).

Chooses the seeds sequentially. The probability of a point to be chosen is
proportional to the minimum cost of assigning it to the existing seeds.

# References
> D. Arthur and S. Vassilvitskii (2007).
> *k-means++: the advantages of careful seeding.*
> 18th Annual ACM-SIAM symposium on Discrete algorithms, 2007.
"""
struct KmppAlg <: SeedingAlgorithm end

function initseeds!(iseeds::IntegerVector, alg::KmppAlg,
                    X::AbstractMatrix{<:Real},
                    metric::PreMetric = SqEuclidean())
    n = size(X, 2)
    k = length(iseeds)
    check_seeding_args(n, k)

    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p

    if k > 1
        mincosts = colwise(metric, X, view(X, :, p))
        mincosts[p] = 0

        # pick remaining (with a chance proportional to mincosts)
        tmpcosts = zeros(n)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p

            # update mincosts
            c = view(X, :, p)
            colwise!(tmpcosts, metric, X, view(X, :, p))
            updatemin!(mincosts, tmpcosts)
            mincosts[p] = 0
        end
    end

    return iseeds
end

function initseeds_by_costs!(iseeds::IntegerVector, alg::KmppAlg,
                             costs::AbstractMatrix{<:Real})
    n = size(costs, 1)
    k = length(iseeds)
    check_seeding_args(n, k)

    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p

    if k > 1
        mincosts = costs[:, p]
        mincosts[p] = 0

        # pick remaining (with a chance proportional to mincosts)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p

            # update mincosts
            updatemin!(mincosts, view(costs, :, p))
            mincosts[p] = 0
        end
    end

    return iseeds
end

"""
    KmCentralityAlg <: SeedingAlgorithm

K-medoids initialization based on centrality (`:kmcen`).

Choose the ``k`` points with the highest *centrality* as seeds.

# References
> Hae-Sang Park and Chi-Hyuck Jun.
> *A simple and fast algorithm for K-medoids clustering.*
> doi:10.1016/j.eswa.2008.01.039
"""
struct KmCentralityAlg <: SeedingAlgorithm end

function initseeds_by_costs!(iseeds::IntegerVector, alg::KmCentralityAlg,
                             costs::AbstractMatrix{<:Real})

    n = size(costs, 1)
    k = length(iseeds)
    check_seeding_args(n, k)

    # compute score for each item
    coefs = vec(sum(costs, dims=2))
    for i = 1:n
        @inbounds coefs[i] = inv(coefs[i])
    end

    # scores[j] = \sum_j costs[i,j] / (\sum_{j'} costs[i,j'])
    #           = costs[i,j] * coefs[i]
    scores = costs'coefs

    # lower score indicates better seeds
    sp = sortperm(scores)
    for i = 1:k
        @inbounds iseeds[i] = sp[i]
    end
    return iseeds
end

initseeds!(iseeds::IntegerVector, alg::KmCentralityAlg, X::AbstractMatrix{<:Real},
           metric::PreMetric = SqEuclidean()) =
    initseeds_by_costs!(iseeds, alg, pairwise(metric, X, dims=2))
