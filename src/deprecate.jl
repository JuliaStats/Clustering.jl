## Deprecated

# deprecated at 0.13
@deprecate kmpp(X, k) initseeds(:kmpp, X, k)
@deprecate kmpp_by_costs(costs, k) initseeds_by_costs(:kmpp, costs, k)

# deprecated at 0.13.1
@deprecate copyseeds(X, iseeds) copyseeds!(Matrix{eltype(X)}(undef, size(X, 1), length(iseeds)), X, iseeds)

# deprecated as of 0.13.2
@deprecate varinfo(k1::Int, a1::AbstractVector{Int},
                   k2::Int, a2::AbstractVector{Int}) varinfo(a1, a2)
@deprecate varinfo(R::ClusteringResult, k0::Int, a0::AbstractVector{Int}) varinfo(R, a0)
