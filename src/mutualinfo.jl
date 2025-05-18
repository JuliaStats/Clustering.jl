# Mutual Information

@inline function _mutualinfo(f, A::AbstractMatrix{<:Integer})
    N = sum(A)
    (N == 0.0) && return 0.0

    rows = sum(A, dims=2)
    cols = sum(A, dims=1)
    entA = entropy(A)
    entArows = entropy(rows)
    entAcols = entropy(cols)

    hck = (entA - entAcols)/N
    hc = entArows/N + log(N)
    hk = entAcols/N + log(N)

    f(hck,hc,hk,rows,cols,N)
end

function _mutualinfo(::Val{T}, a, b; kwargs...) where T
    if T === :adjusted && isempty(kwargs)
        error("Error: mutualinfo(): `method=:adjusted` requires SpecialFunctions package to be loaded")
    elseif T !== :classic && T !== :normalized
        throw(ArgumentError("mutualinfo(): `method=:$(T)` is not supported"))
    else
        throw(ArgumentError("mutualinfo(): unsupported kwargs used. See the `mutualinfo` docstring for more information"))
    end
end

"""
    mutualinfo(a, b; method=:normalized, kwargs...) -> Float64

Compute the *mutual information* between the two clusterings of the same
data points.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

`method` can be one of `:classic`, `:normalized` (default), or `:adjusted`, to calculate the
original mutual information score, the normalized mutual information, or the adjusted mutual
information respectively.

When `method=:adjusted`, the `aggregate` kwarg determines how the normalizer
in the denominator is computed. It can be one of:
- `:mean`: The arithmetic mean of two values
- `:geomean`:  The geometric mean of two values
- `:max`: The highest of two values
- `:min`: The lowest of two values

# References
> Vinh, Epps, and Bailey, (2009). *Information theoretic measures for clusterings comparison*.
> Proceedings of the 26th Annual International Conference on Machine Learning - ICML ‘09.

> "Data Mining Practical Machine Tools and Techniques", Witten & Frank 2005.
"""
function mutualinfo(a, b; method::Union{Nothing, Symbol} = nothing, normed::Union{Nothing, Bool} = nothing, kwargs...)
    # Disallow `method` and `normed` to be used together
    if method === nothing
        (normed === nothing) || Base.depwarn("`normed` kwarg is deprecated, please use `method=:normalized` instead of `normed=true`, and `method=:classic` instead of `normed=false'", :mutualinfo)
        method = if (normed === nothing) || normed
            :normalized
        else
            :classic
        end
    else
        (normed === nothing) || throw(ArgumentError("`normed` kwarg is not compatible with `method` kwarg"))
    end
    # Little hack to ensure the invalid kwargs error is thrown
    if method === :adjusted && length(kwargs) >= 1 && :aggregate ∉ keys(kwargs)
        method = :classic
    end

    _mutualinfo(Val(method), a, b; kwargs...)
end

function _mutualinfo(::Val{:normalized}, a, b)
    return _mutualinfo(counts(a, b)) do hck, hc, hk, _, _, _
        mi = hc - hck
        2*mi/(hc+hk)
    end
end
function _mutualinfo(::Val{:classic}, a, b)
    return _mutualinfo(counts(a, b)) do hck, hc, _, _, _, _
        hc - hck
    end
end
