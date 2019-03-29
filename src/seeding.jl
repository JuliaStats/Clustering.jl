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
Base type for all seeding algorithms.

Each seeding algorithm should implement the two functions: [`initseeds!`](@ref)
and [`initseeds_by_costs!`](@ref).
"""
abstract type SeedingAlgorithm end

"""
    initseeds(alg::Union{SeedingAlgorithm, Symbol},
              X::AbstractMatrix, k::Integer)

Select `k` seeds from a ``d×n`` data matrix `X` using the `alg`
algorithm.

`alg` could be either an instance of [`SeedingAlgorithm`](@ref) or a symbolic
name of the algorithm.

Returns an integer vector of length `k` that contains the indices of
chosen seeds.
"""
initseeds(alg::SeedingAlgorithm, X::AbstractMatrix{<:Real}, k::Integer) =
    initseeds!(Vector{Int}(undef, k), alg, X)

"""
    initseeds_by_costs(alg::Union{SeedingAlgorithm, Symbol},
                       costs::AbstractMatrix, k::Integer)

Select `k` seeds from the ``n×n`` `costs` matrix using algorithm `alg`.

Here, ``\\mathrm{costs}_{ij}`` is the cost of assigning points ``i`` and ``j``
to the same cluster. One may, for example, use the squared Euclidean distance
between the points as the cost.

Returns an integer vector of length `k` that contains the indices of
chosen seeds.
"""
initseeds_by_costs(alg::SeedingAlgorithm, costs::AbstractMatrix{<:Real}, k::Integer) =
    initseeds_by_costs!(Vector{Int}(undef, k), alg, costs)

seeding_algorithm(s::Symbol) =
    s == :rand ? RandSeedAlg() :
    s == :kmpp ? KmppAlg() :
    s == :kmcen ? KmCentralityAlg() :
    error("Unknown seeding algorithm $s")

initseeds(algname::Symbol, X::AbstractMatrix{<:Real}, k::Integer) =
    initseeds(seeding_algorithm(algname), X, k)::Vector{Int}

initseeds_by_costs(algname::Symbol, costs::AbstractMatrix{<:Real}, k::Integer) =
    initseeds_by_costs(seeding_algorithm(algname), costs, k)

initseeds(iseeds::Vector{Int}, X::AbstractMatrix{<:Real}, k::Integer) = iseeds
initseeds_by_costs(iseeds::Vector{Int}, costs::AbstractMatrix{<:Real}, k::Integer) = iseeds

function copyseeds!(S::Matrix{<:AbstractFloat}, X::AbstractMatrix{<:Real},
                    iseeds::AbstractVector)
    d, n = size(X)
    k = length(iseeds)
    size(S) == (d, k) || throw(DimensionMismatch("Inconsistent array dimensions."))
    for j = 1:k
        copyto!(view(S, :, j), view(X, :, iseeds[j]))
    end
    return S
end

# NOTE: this should eventually be removed as only `copyseeds!` is used in `kmeans`.
copyseeds(X::AbstractMatrix{<:Real}, iseeds::AbstractVector) =
    copyseeds!(Matrix{eltype(X)}(undef, size(X, 1), length(iseeds)), X, iseeds)

function check_seeding_args(n::Integer, k::Integer)
    k >= 1 || error("The number of seeds must be positive.")
    k <= n || error("Attempted to select more seeds than data points.")
end

"""
Random seeding (`:rand`).

Chooses an arbitrary subset of ``k`` data points as cluster seeds.
"""
struct RandSeedAlg <: SeedingAlgorithm end

"""
    initseeds!(iseeds::AbstractVector{Int}, alg::SeedingAlgorithm,
               X::AbstractMatrix)

Initialize `iseeds` with the indices of cluster seeds for the `X` data matrix
using the `alg` seeding algorithm.

Returns `iseeds`.
"""
initseeds!(iseeds::IntegerVector, alg::RandSeedAlg, X::AbstractMatrix{<:Real}) =
    sample!(1:size(X, 2), iseeds; replace=false)

"""
    initseeds_by_costs!(iseeds::AbstractVector{Int}, alg::SeedingAlgorithm,
                        costs::AbstractMatrix)

Initialize `iseeds` with the indices of cluster seeds for the `costs` matrix
using the `alg` seeding algorithm.

Here, ``\\mathrm{costs}_{ij}`` is the cost of assigning points ``i`` and ``j``
to the same cluster. One may, for example, use the squared Euclidean distance
between the points as the cost.

Returns `iseeds`.
"""
initseeds_by_costs!(iseeds::IntegerVector, alg::RandSeedAlg, X::AbstractMatrix{<:Real}) =
    sample!(1:size(X,2), iseeds; replace=false)

"""
Kmeans++ seeding (`:kmpp`).

Chooses the seeds sequentially. The probability of a point to be chosen is
proportional to the minimum cost of assigning it to the existing seeds.

> D. Arthur and S. Vassilvitskii (2007).
> *k-means++: the advantages of careful seeding.*
> 18th Annual ACM-SIAM symposium on Discrete algorithms, 2007.
"""
struct KmppAlg <: SeedingAlgorithm end

function initseeds!(iseeds::IntegerVector, alg::KmppAlg,
                    X::AbstractMatrix{<:Real}, metric::PreMetric)
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

initseeds!(iseeds::IntegerVector, alg::KmppAlg, X::AbstractMatrix{<:Real}) =
    initseeds!(iseeds, alg, X, SqEuclidean())

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
K-medoids initialization based on centrality (`:kmcen`).

Choose the ``k`` points with the highest *centrality* as seeds.

> Hae-Sang Park and Chi-Hyuck Jun.
> *A simple and fast algorithm for K-medoids clustering.*
> doi:10.1016/j.eswa.2008.01.039
"""
struct KmCentralityAlg <: SeedingAlgorithm end

function initseeds_by_costs!(iseeds::IntegerVector, alg::KmCentralityAlg,
                             costs::AbstractMatrix{<:Real})

    n = size(costs, 1)
    k = length(iseeds)
    k <= n || error("Attempted to select more seeds than points.")

    # compute score for each item
    coefs = vec(sum(costs, dims=2))
    for i = 1:n
        @inbounds coefs[i] = inv(coefs[i])
    end

    # scores[j] = \sum_j costs[i,j] / (\sum_{j'} costs[i,j'])
    #           = costs[i,j] * coefs[i]
    #
    # So this is matrix-vector multiplication
    scores = costs'coefs

    # lower score indicates better seeds
    sp = sortperm(scores)
    for i = 1:k
        @inbounds iseeds[i] = sp[i]
    end
    return iseeds
end

initseeds!(iseeds::IntegerVector, alg::KmCentralityAlg, X::AbstractMatrix{<:Real}, metric::PreMetric) =
    initseeds_by_costs!(iseeds, alg, pairwise(metric, X, dims=2))

initseeds!(iseeds::IntegerVector, alg::KmCentralityAlg, X::AbstractMatrix{<:Real}) =
    initseeds!(iseeds, alg, X, SqEuclidean())
