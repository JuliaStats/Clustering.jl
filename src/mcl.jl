# MCL (Markov CLustering algorithm)

"""
    immutable MCLResult <: ClusteringResult

Result returned by `mcl()`.
"""
immutable MCLResult <: ClusteringResult
    mcl_adj::Matrix{Float64}    # final MCL adjacency matrix (equailibrium state matrix if converged)
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
function _mcl_clusters(mcl_adj::Matrix{Float64}, allow_singles::Bool, zero_tol::Float64 = 1E-20)
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

# adjacency matrix expansion (matrix-wise raising to a given integer power) kernel
# FIXME `_mcl_expand!()` that does not allocate new expanded matrix
function _mcl_expand(src::Matrix, expansion::Integer)
    src ^ expansion
end

# adjacency matrix expansion (matrix-wise raising to a given power) kernel
# FIXME `_mcl_expand!()` that does not allocate new expanded matrix
function _mcl_expand(src::Matrix, expansion::Number)
    # we have to implement the workarond for matrix-power because of julia bug #16930
    isinteger(expansion) && return _mcl_expand(src, Integer(real(expansion)))
    v, X = eig(src)
    # FIXME type instability here, revisit when #16930 is fixed
    (eltype(v) <: Complex) || (any(v.<0) && (v = complex(v)))
    Xinv = ishermitian(src) ? X' : inv(X)
    X * Diagonal(v.^expansion) * Xinv
end

# adjacency matrix inflation (element-wise raising to a given power) kernel
function _mcl_inflate!(dest::Matrix{Float64}, src::Matrix{Complex128}, inflation::Number)
    src_norm = vecnorm(src)
    min_rel = -1E-3*src_norm
    min_img = 1E-3*src_norm
    @inbounds for (i, el) in enumerate(src)
        rel = real(el)
        img = imag(el)
        if rel < min_rel || (abs(img) > min_img && abs(img) > 1E-3*abs(rel))
            throw(InexactError())
        end
        dest[i] = max(0.0, rel)^inflation
    end
    return dest
end

# adjacency matrix inflation (element-wise raising to a given power) kernel
function _mcl_inflate!(dest::Matrix{Float64}, src::Matrix{Float64}, inflation::Number)
    any(src .< -1E-3*vecnorm(src)) && throw(InexactError())
    map!(el -> max(0.0, el)^inflation, dest, src)
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
* `display::Symbol`: `:none` for no output or `:verbose` for diagnostic messages

See [original MCL implementation](http://micans.org/mcl).
"""
@compat function mcl(adj::Matrix{Float64};
             add_loops::Bool = true,
             expansion::Number = 2, inflation::Number = 2.0,
             save_final_matrix::Bool = false,
             allow_singles::Bool = true,
             max_iter::Integer = 100, tol::Number=1E-5,
             display::Symbol=:none)#::MCLResult FIXME uncomment when 0.4 support is dropped

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

    MCLResult(save_final_matrix ? mcl_adj : Matrix{Float64}(0,0),
              el2clu, clu_sizes, nunassigned,
              niter, rel_delta, converged)
end
