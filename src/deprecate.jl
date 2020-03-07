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

# FIXME remove after deprecation period for merge/labels/height/method
Base.propertynames(hclu::Hclust, private::Bool = false) =
    (fieldnames(typeof(hclu))...,
     #= deprecated as of 0.12 =# :height, :labels, :merge, :method)

# FIXME remove after deprecation period for merge/labels/height/method
@inline function Base.getproperty(hclu::Hclust, prop::Symbol)
    if prop === :height # deprecated as of 0.12
        Base.depwarn("Hclust::height is deprecated, use Hclust::heights", Symbol("Hclust::height"))
        return getfield(hclu, :heights)
    elseif prop === :labels # deprecated as of 0.12
        Base.depwarn("Hclust::labels is deprecated and will be removed in future versions", Symbol("Hclust::labels"))
        return 1:nnodes(hclu)
    elseif prop === :merge # deprecated as of 0.12
        Base.depwarn("Hclust::merge is deprecated, use Hclust::merges", Symbol("Hclust::merge"))
        return getfield(hclu, :merges)
    elseif prop === :method # deprecated as of 0.12
        Base.depwarn("Hclust::method is deprecated, use Hclust::linkage", Symbol("Hclust::method"))
        return getfield(hclu, :linkage)
    else
        return getfield(hclu, prop)
    end
end

# FIXME remove after deprecation period for cweights
Base.propertynames(clu::KmeansResult, private::Bool = false) =
    (fieldnames(typeof(clu))..., #= deprecated as of 0.13.2 =# :cweights)

# FIXME remove after deprecation period for cweights
@inline function Base.getproperty(clu::KmeansResult, prop::Symbol)
    if prop === :cweights # deprecated as of 0.13.2
        Base.depwarn("KmeansResult::cweights is deprecated, use wcounts(clu::KmeansResult)",
                     Symbol("KmeansResult::cweights"))
        return clu.wcounts
    else
        return getfield(clu, prop)
    end
end

# FIXME remove after deprecation period for acosts
Base.propertynames(kmed::KmedoidsResult, private::Bool = false) =
    (fieldnames(typeof(kmed))..., #= deprecated since v0.13.4=# :acosts)

# FIXME remove after deprecation period for acosts
function Base.getproperty(kmed::KmedoidsResult, prop::Symbol)
    if prop == :acosts # deprecated since v0.13.4
        Base.depwarn("KmedoidsResult::acosts is deprecated, use KmedoidsResult::costs",
                     Symbol("KmedoidsResult::costs"))
        return getfield(kmed, :costs)
    else
        return getfield(kmed, prop)
    end
end
