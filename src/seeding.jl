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

abstract SeedingAlgorithm

initseeds(alg::SeedingAlgorithm, X::RealMatrix, k::Integer) = 
    initseeds!(Array(Int, k), alg, X)

initseeds_by_costs(alg::SeedingAlgorithm, costs::RealMatrix, k::Integer) = 
    initseeds_by_costs!(Array(Int, k), alg, costs)

seeding_algorithm(s::Symbol) = 
    s == :rand ? RandSeedAlg() :
    s == :kmpp ? KmppAlg() :
    s == :kmcen ? KmCentralityAlg() :
    error("Unknown seeding algorithm $s")

initseeds(algname::Symbol, X::RealMatrix, k::Integer) = 
    initseeds(seeding_algorithm(algname), X, k)::Vector{Int}

initseeds_by_costs(algname::Symbol, costs::RealMatrix, k::Integer) = 
    initseeds_by_costs(seeding_algorithm(algname), costs, k)

initseeds(iseeds::Vector{Int}, X::RealMatrix, k::Integer) = iseeds
initseeds_by_costs(iseeds::Vector{Int}, costs::RealMatrix, k::Integer) = iseeds

function copyseeds!(S::DenseMatrix, X::DenseMatrix, iseeds::AbstractVector)
    d = size(X, 1)
    n = size(X, 2)
    k = length(iseeds)
    (size(X,1) == d && size(S) == (d, k)) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    for j = 1:k
        copy!(view(S,:,j), view(X,:,iseeds[j]))
    end
    return S
end

copyseeds{T}(X::DenseMatrix{T}, iseeds::AbstractVector) = 
    copyseeds!(Array(T, size(X,1), length(iseeds)), X, iseeds)

function check_seeding_args(n::Integer, k::Integer)
    k >= 1 || error("The number of seeds must be positive.")
    k <= n || error("Attempted to select more seeds than samples.")
end


# Random seeding
#
#   choose an arbitrary subset as seeds
#

type RandSeedAlg <: SeedingAlgorithm end

initseeds!(iseeds::IntegerVector, alg::RandSeedAlg, X::RealMatrix) = 
    sample!(1:size(X,2), iseeds; replace=false)

initseeds_by_costs!(iseeds::IntegerVector, alg::RandSeedAlg, X::RealMatrix) = 
    sample!(1:size(X,2), iseeds; replace=false)


# Kmeans++ seeding
#
#   D. Arthur and S. Vassilvitskii (2007). 
#   k-means++: the advantages of careful seeding. 
#   18th Annual ACM-SIAM symposium on Discrete algorithms, 2007.
#

type KmppAlg <: SeedingAlgorithm end

function initseeds!(iseeds::IntegerVector, alg::KmppAlg, X::RealMatrix, metric::PreMetric)
    n = size(X, 2)
    k = length(iseeds)
    check_seeding_args(n, k)

    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p

    if k > 1
        mincosts = colwise(metric, X, view(X,:,p))
        mincosts[p] = 0

        # pick remaining (with a chance proportional to mincosts)
        tmpcosts = zeros(n)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p

            # update mincosts
            c = view(X,:,p)
            colwise!(tmpcosts, metric, X, view(X,:,p))
            updatemin!(mincosts, tmpcosts)
            mincosts[p] = 0
        end
    end

    return iseeds
end

initseeds!(iseeds::IntegerVector, alg::KmppAlg, X::RealMatrix) = 
    initseeds!(iseeds, alg, X, SqEuclidean())

function initseeds_by_costs!(iseeds::IntegerVector, alg::KmppAlg, costs::RealMatrix)
    n = size(costs, 1)
    k = length(iseeds)
    check_seeding_args(n, k)

    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p

    if k > 1
        mincosts = copy(view(costs,:,p))
        mincosts[p] = 0

        # pick remaining (with a chance proportional to mincosts)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p

            # update mincosts
            updatemin!(mincosts, view(costs,:,p))
            mincosts[p] = 0
        end
    end

    return iseeds
end

kmpp(X::RealMatrix, k::Int) = initseeds(KmppAlg(), X, k)
kmpp_by_costs(costs::RealMatrix, k::Int) = initseeds(KmppAlg(), costs, k)


# K-medoids initialization based on centrality
#
#   Hae-Sang Park and Chi-Hyuck Jun.
#   A simple and fast algorithm for K-medoids clustering.
#   doi:10.1016/j.eswa.2008.01.039
#

type KmCentralityAlg <: SeedingAlgorithm end

function initseeds_by_costs!(iseeds::IntegerVector, alg::KmCentralityAlg, costs::RealMatrix)
    n = size(costs, 1)
    k = length(iseeds)
    k <= n || error("Attempted to select more seeds than samples.")

    # compute score for each item
    coefs = vec(sum(costs, 2))
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

initseeds!(iseeds::IntegerVector, alg::KmCentralityAlg, X::RealMatrix, metric::PreMetric) = 
    initseeds_by_costs!(iseeds, alg, pairwise(metric, X))

initseeds!(iseeds::IntegerVector, alg::KmCentralityAlg, X::RealMatrix) = 
    initseeds!(iseeds, alg, X, SqEuclidean())


