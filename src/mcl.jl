# MCL (Markov CLustering algorithm)

"""
    immutable MCLResult <: ClusteringResult

Result returned by `mcl()`.
"""
immutable MCLResult <: ClusteringResult
    mcl_adj::AbstractMatrix     # final MCL adjacency matrix (equilibrium state matrix if converged)
    assignments::Vector{Int}    # element-to-cluster assignments (n)
    counts::Vector{Int}         # number of samples assigned to each cluster (k)
    nunassigned::Int            # number of single elements not assigned to any cluster
    iterations::Int             # number of elapsed iterations
    rel_Δ::Float64              # final relative Δ
    converged::Bool             # whether the procedure converged
end

# Extract clusters from the final (equilibrium) MCL matrix
# Return the tuple: cluster indices for each element, cluster sizes,
# the number of unassigned (0 cluster index) elements (if `allow_singles` is on)
# `zero_tol` is a minimal value to consider as an element-to-cluster assignment
function _mcl_clusters(mcl_adj::AbstractMatrix, allow_singles::Bool, zero_tol::Float64 = 1E-20)
    # remove rows containing only zero elements and convert into a mask of nonzero elements
    el2clu_mask = mcl_adj[squeeze(sum(mcl_adj, 2), 2) .> zero_tol, :] .> zero_tol

    # assign cluster indexes to each node
    # cluster index is the index of the first TRUE in a given column
    clu_ixs = squeeze(mapslices(el_mask -> !isempty(el_mask) ? findmax(el_mask)[2] : 0,
                                el2clu_mask, 1), 1)
    clu_sizes = zeros(Int, size(el2clu_mask, 1))
    unassigned_count = 0
    @inbounds for clu_ix in clu_ixs
        (clu_ix > 0) && (clu_sizes[clu_ix] += 1)
    end
    if !allow_singles
        # collapse all size 1 clusters into one with index 0
        @inbounds for i in eachindex(clu_ixs)
            clu_ix = clu_ixs[i]
            if clu_ix > 0 && clu_sizes[clu_ix] == 1
                clu_ixs[i] = 0
                clu_sizes[clu_ix] = 0
                unassigned_count += 1
            end
        end
    else
        unassigned_count = 0
    end

    # recode clusters numbers to be in 1:N range (or 0:N if there's collapsed cluster)
    clu_id_map = zeros(Int, length(clu_sizes))
    next_clu_ix = 0
    @inbounds for i in eachindex(clu_ixs)
        old_clu_ix = clu_ixs[i]
        if old_clu_ix > 0
            new_clu_ix = clu_id_map[old_clu_ix]
            clu_ixs[i] = new_clu_ix == 0 ?
                         clu_id_map[old_clu_ix] = (next_clu_ix += 1) :
                         new_clu_ix
        end
    end
    old_clu_sizes = clu_sizes
    clu_sizes = zeros(Int, next_clu_ix)
    for (old_clu_ix, new_clu_ix) in enumerate(clu_id_map)
        if new_clu_ix > 0
            clu_sizes[new_clu_ix] = old_clu_sizes[old_clu_ix]
        end
    end

    clu_ixs, clu_sizes, unassigned_count
end

# adjacency matrix expansion (matrix-wise raising to a given power) kernel
function _mcl_expand(src::AbstractMatrix, expansion::Number)
    try
        return src^expansion
    catch ex # FIXME: remove this when functionality become available in the standard library
        if isa(ex, MethodError)
            throw(ArgumentError("MCL expansion of $(typeof(src)) with expansion=$expansion not supported"))
        else
            rethrow()
        end
    end
end

# integral power inflation (single matrix element)
_mcl_el_inflate(el::Number, inflation::Integer) = el^inflation

# non-integral power inflation (single matrix element)
_mcl_el_inflate(el::Number, inflation::Number) = real((el+0im)^inflation)

# adjacency matrix inflation (element-wise raising to a given power) kernel
function _mcl_inflate!(dest::AbstractMatrix, src::AbstractMatrix, inflation::Number)
    @inbounds for i in eachindex(src)
        dest[i] = _mcl_el_inflate(src[i], inflation)
    end
end

# adjacency matrix pruning
function _mcl_prune!(mtx::AbstractMatrix, prune_tol::Number)
    for i in 1:size(mtx,2)
        c = view(mtx, :, i)
        θ = mean(c)*prune_tol
        @inbounds @simd for j in eachindex(c)
            c[j] = ifelse(c[j] >= θ, c[j], 0.0)
        end
    end
    issparse(mtx) && dropzeros!(mtx)
    return mtx
end

"""
    mcl(adj::Matrix; [keyword arguments])::MCLResult

Identify clusters in the weighted graph using Markov Clustering Algorithm (MCL).

# Arguments
* `adj::Matrix{Float64}`: adjacency matrix that defines the weighted graph
  to cluster
* `add_loops::Bool`: whether edges of weight 1.0 from the node to itself
  should be appended to the graph (enabled by default)
* `expansion::Number`: MCL expansion constant (2)
* `inflation::Number`: MCL inflation constant (2.0)
* `save_final_matrix::Bool`: save final equilibrium state in the result,
  otherwise leave it empty; disabled by default, could be useful if
  MCL doesn't converge
* `max_iter::Integer`: max number of MCL iterations
* `tol::Number`: MCL adjacency matrix convergence threshold
* `prune_tol::Number`: pruning threshold
* `display::Symbol`: `:none` for no output or `:verbose` for diagnostic messages

See [original MCL implementation](http://micans.org/mcl).

Ref: Stijn van Dongen, "Graph clustering by flow simulation", 2001
"""
function mcl{T<:Real}(adj::AbstractMatrix{T};
                      add_loops::Bool = true,
                      expansion::Number = 2, inflation::Number = 2.0,
                      save_final_matrix::Bool = false,
                      allow_singles::Bool = true,
                      max_iter::Integer = 100, tol::Number=1.0e-5,
                      prune_tol::Number=1.0e-5, display::Symbol=:none)
    m, n = size(adj)
    m == n || throw(DimensionMismatch("Square adjacency matrix expected"))

    if add_loops
        @inbounds for i in 1:size(adj, 1)
            adj[i, i] = 1.0
        end
    end

    # initialize the MCL adjacency matrix by normalized `adj` weights
    mcl_adj = copy(adj)
    # normalize in columns
    scale!(mcl_adj, map(x -> x != 0.0 ?  1.0/x : x, squeeze(sum(mcl_adj, 1), 1)))
    mcl_norm = vecnorm(mcl_adj)
    if !isfinite(mcl_norm)
        throw(OverflowError("The norm of the input adjacency matrix is not finite"))
    end
    next_mcl_adj = similar(mcl_adj)

    # do MCL iterations
    if display != :none
        info("Starting MCL iterations...")
    end
    niter = 0
    converged = false
    rel_delta = NaN
    while !converged && niter < max_iter
        expanded = _mcl_expand(mcl_adj, expansion)
        _mcl_inflate!(next_mcl_adj, expanded, inflation)
        _mcl_prune!(next_mcl_adj, prune_tol)

        # normalize in columns
        scale!(next_mcl_adj, map(x -> x != 0.0 ? 1.0/x : x,
                                 squeeze(sum(next_mcl_adj, 1), 1)))

        next_mcl_norm = vecnorm(next_mcl_adj)
        if !isfinite(next_mcl_norm)
            warn("MCL adjacency matrix norm is not finite")
            break
        end
        rel_delta = euclidean(next_mcl_adj, mcl_adj)/mcl_norm
        (display == :verbose) && info("MCL iter. #$niter: rel.Δ=", rel_delta)
        (converged = rel_delta <= tol) && break
        # update (swap) MCL adjacency
        niter += 1
        mcl_adj, next_mcl_adj = next_mcl_adj, mcl_adj
        mcl_norm = next_mcl_norm
        (mcl_norm < tol) && break # matrix is zero
    end

    if display != :none
        if converged
            info("MCL converged after $niter iteration(s)")
        else
            warn("MCL didn't converge after $niter iteration(s)")
        end
    end

    (display == :verbose) && info("Generating MCL clusters...")
    el2clu, clu_sizes, nunassigned = _mcl_clusters(mcl_adj, allow_singles,
                                                   tol/length(mcl_adj))

    return MCLResult(save_final_matrix ? mcl_adj : similar(mcl_adj, (0,0)),
              el2clu, clu_sizes, nunassigned, niter, rel_delta, converged)
end
