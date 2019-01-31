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
#       - X:        the data matrix, each column being a sample
#       - costs:    pre-computed pairwise cost matrix.
#
#   This function returns iseeds
#

abstract type SeedingAlgorithm end

initseeds(alg::SeedingAlgorithm, X::AbstractMatrix{<:Real}, k::Integer) =
    initseeds!(Vector{Int}(undef, k), alg, X)

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

function copyseeds!(S::Matrix{TS}, X::AbstractMatrix{TX},
                    iseeds::AbstractVector) where {TS<:AbstractFloat, TX<:Real}
    d, n = size(X)
    k = length(iseeds)
    size(S) == (d, k) || throw(DimensionMismatch("Inconsistent array " *
                                                 " dimensions."))
    for j = 1:k
        copyto!(view(S, :, j), view(X, :, iseeds[j]))
    end
    return S
end

copyseeds(X::AbstractMatrix{<:Real}, iseeds::AbstractVector) =
    copyseeds!(Matrix{eltype(X)}(undef, size(X, 1), length(iseeds)), X, iseeds)

function check_seeding_args(n::Integer, k::Integer)
    k >= 1 || error("The number of seeds must be positive.")
    k <= n || error("Attempted to select more seeds than samples.")
end


# Random seeding
#
#   choose an arbitrary subset as seeds
#

struct RandSeedAlg <: SeedingAlgorithm end

initseeds!(iseeds::IntegerVector, alg::RandSeedAlg, X::AbstractMatrix{<:Real}) = sample!(1:size(X, 2), iseeds; replace=false)

initseeds_by_costs!(iseeds::IntegerVector, alg::RandSeedAlg, X::AbstractMatrix{<:Real}) = sample!(1:size(X,2), iseeds; replace=false)


# Kmeans++ seeding
#
#   D. Arthur and S. Vassilvitskii (2007).
#   k-means++: the advantages of careful seeding.
#   18th Annual ACM-SIAM symposium on Discrete algorithms, 2007.
#

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

kmpp(X::AbstractMatrix{<:Real}, k::Int) = initseeds(KmppAlg(), X, k)
kmpp_by_costs(costs::AbstractMatrix{<:Real}, k::Int) = initseeds(KmppAlg(), costs, k)


# K-medoids initialization based on centrality
#
#   Hae-Sang Park and Chi-Hyuck Jun.
#   A simple and fast algorithm for K-medoids clustering.
#   doi:10.1016/j.eswa.2008.01.039
#

struct KmCentralityAlg <: SeedingAlgorithm end

function initseeds_by_costs!(iseeds::IntegerVector, alg::KmCentralityAlg,
                             costs::AbstractMatrix{<:Real})

    n = size(costs, 1)
    k = length(iseeds)
    k <= n || error("Attempted to select more seeds than samples.")

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
    initseeds_by_costs!(iseeds, alg, pairwise(metric, X))

initseeds!(iseeds::IntegerVector, alg::KmCentralityAlg, X::AbstractMatrix{<:Real}) =
    initseeds!(iseeds, alg, X, SqEuclidean())
