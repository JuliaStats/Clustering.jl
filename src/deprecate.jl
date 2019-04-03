## Deprecated

# deprecated at 0.13
@deprecate kmpp(X, k) initseeds(:kmpp, X, k)
@deprecate kmpp_by_costs(costs, k) initseeds_by_costs(:kmpp, costs, k)

# deprecated at 0.13.1
@deprecate copyseeds(X, iseeds) copyseeds!(Matrix{eltype(X)}(undef, size(X, 1), length(iseeds)), X, iseeds)
