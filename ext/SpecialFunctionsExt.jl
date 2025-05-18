module SpecialFunctionsExt # Should be same name as the file (just like a normal package)

using SpecialFunctions: loggamma
using StatsBase: counts
using Statistics: middle

import Clustering: _mutualinfo

function _mutualinfo(::Val{:adjusted}, a, b; aggregate::Symbol = :mean)
    norm_f = if aggregate === :mean
        middle
    elseif aggregate === :geomean
        (a,b) -> sqrt(a*b)
    elseif aggregate === :max
        max
    elseif aggregate === :min
        min
    else
        throw(ArgumentError("Valid options for `aggregate` are: `mean`, `geomean`, `max`, `min`"))
    end

    return _mutualinfo(counts(a, b)) do hck, hc, hk, rows, cols, N
        mi = hc - hck
        emi = _expectedmutualinfo(rows, cols, N)
        normalizer = norm_f(hc, hk)
        denominator = normalizer - emi
        (mi - emi) / denominator
    end
end

# Adjusted Mutual Information

function _expectedmutualinfo(a, b, n_samples)
    nijs = 1:max(maximum(a), maximum(b))

    term1 = nijs ./ n_samples

    log_ab = [log(a[i]) + log(b[j]) for i in eachindex(a), j in eachindex(b)]
    log_Nnij = log(n_samples) .+ log.(nijs)

    gln_a = loggamma.(a .+ 1)
    gln_b = loggamma.(b .+ 1)
    gln_Na = loggamma.(n_samples .- a .+ 1)
    gln_Nb = loggamma.(n_samples .- b .+ 1)
    gln_Nnij = loggamma.(nijs .+ 1) .+ loggamma.(n_samples + 1)

    emi = zero(Float64)
    for i in eachindex(a), j in eachindex(b)
        nij_idxs = max(1, a[i] - n_samples + b[j]):min(a[i], b[j])
        for nij in nij_idxs
            term2 = log_Nnij[nij] - log_ab[i,j]
            gln = (
                gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j] - gln_Nnij[nij] -
                loggamma(a[i] - nij + 1) - loggamma(b[j] - nij + 1) -
                loggamma(n_samples - a[i] - b[j] + nij + 1)
            )
            term3 = exp(gln)
            emi += (term1[nij] * term2 * term3)
        end
    end
    return emi
end

end # module
