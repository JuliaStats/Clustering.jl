## Deprecated


## old initialization algorithms
#
# deprecated at 0.2.7
#

function randseed_initialize!{T<:AbstractFloat}(X::Matrix{T}, centers::Matrix{T})
    Base.depwarn("randseed_initialize! is deprecated. Please use initseeds! instead.", 
                 :randseed_initialize!)

    K = size(centers, 2)
    iseeds = initseeds(RandSeedAlg(), X, K)
    copyseeds!(centers, X, iseeds)
end

function kmeanspp_initialize!{T<:AbstractFloat}(X::Matrix{T}, centers::Matrix{T})
    Base.depwarn("kmeanspp_initialize! is deprecated. Please use initseeds! instead.", 
                 :kmeanspp_initialize!)

    K = size(centers, 2)
    iseeds = initseeds(KmppAlg(), X, K)
    copyseeds!(centers, X, iseeds)
end

function initial_medoids{T<:AbstractFloat}(costs::Matrix{T}, k::Int)
    Base.depwarn("initial_medoids is deprecated. Please use initseeds_by_costs instead.", 
                 :initial_medoids)

    initseeds_by_costs(KmCentralityAlg(), costs, k)
end

function affinity_propagation(x::Matrix; opts...)
    Base.depwarn("affinity_propagation is deprecated. Please use affinityprop instead.", 
                 :affinity_propagation)
    affinityprop(x; opts...)
end

