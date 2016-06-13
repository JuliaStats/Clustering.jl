# MCL (Markov CLustering algorithm)

#### Interface

"""
    `mcl()` result.
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

""" _mcl_clusters(mcl_adj, allow_singles, zero_tol = 1E-20)

    Extract clusters from the final MCL matrix `mcl_adj`.
    Returns the vector of cluster indices for each element, the sizes of the
    clusters and the number of the unassigned elements (`allow_singles` controls
    whether 1-element clusters would be retained or collapsed into the set of
    unassigned elements with the cluster index 0).

    See also `mcl()`.
"""
function _mcl_clusters(mcl_adj::Matrix{Float64}, allow_singles::Bool, zero_tol::Float64 = 1E-20)
    # remove rows containing only zero elements and convert into a mask of nonzero elements
    el2clu_mask = mcl_adj[squeeze(sum(mcl_adj, 2), 2) .> zero_tol, :] .> zero_tol

    # assign cluster indexes to each node
    # cluster index is the index of the first TRUE in a given column
    clu_ixs = squeeze(mapslices(el_mask -> findmax(el_mask)[2], el2clu_mask, 1), 1)
    clu_sizes = zeros(Int, size(el2clu_mask, 1))
    unassigned_count = 0
    @inbounds for clu_ix in clu_ixs
        clu_sizes[clu_ix] += 1
    end
    if !allow_singles
        # collapse all size 1 clusters into one with index 0
        @inbounds for i in eachindex(clu_ixs)
            clu_ix = clu_ixs[i]
            if clu_sizes[clu_ix] == 1
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
            if new_clu_ix == 0
                next_clu_ix += 1
                clu_ixs[i] = clu_id_map[old_clu_ix] = next_clu_ix
            else
                clu_ixs[i] = new_clu_ix
            end
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

""" mcl(adj::Matrix)

    Identifies clusters in the weighted graph specified by its adjacency
    matrix using Markov Clustering Algorithm (MCL).

    Returns `MCLResult`.

    See http://micans.org/mcl/
"""
function mcl(adj::Matrix{Float64};
             add_loops::Bool = true,
             expansion::Number = 2, inflation::Number = 2.0,
             allow_singles::Bool = true,
             max_iter = 100, tol::Number=1E-5,
             display::Symbol=:none)

    m, n = size(adj)
    m == n || throw(ArgumentError("Square adjacency matrix expected"))

    if add_loops
        @inbounds for i in 1:size(adj, 1)
            adj[i, i] = 1.0
        end
    end

    # initialize the MCL adjacency matrix by normalized `adj` weights
    mcl_adj = copy(adj)
    # normalize in columns
    scale!(mcl_adj, 1.0./squeeze(sum(mcl_adj, 1), 1))
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
        # expand (raise to the matrix power)
        next_mcl_adj = mcl_adj ^ expansion # FIXME don't reallocate
        # inflate (apply power element-wise)
        @inbounds for i in eachindex(next_mcl_adj)
            next_mcl_adj[i] ^= inflation
        end
        # normalize in columns
        scale!(next_mcl_adj, 1.0./squeeze(sum(next_mcl_adj, 1), 1))

        next_mcl_norm = vecnorm(next_mcl_adj)
        if !isfinite(next_mcl_norm)
            warn("MCL adjacency matrix norm is not finite")
            break
        end
        rel_delta = euclidean(next_mcl_adj, mcl_adj)/mcl_norm
        if display == :verbose
            info("MCL iter. #$niter: rel.Δ=", rel_delta)
        end
        converged = rel_delta <= tol
        if converged break end
        # update (swap) MCL adjacency
        niter += 1
        mcl_adj, next_mcl_adj = next_mcl_adj, mcl_adj
        mcl_norm = next_mcl_norm
    end

    if display != :none
        if converged
            info("MCL converged after $niter iteration(s)")
        else
            warn("MCL didn't converge after $niter iteration(s)")
        end
    end

    if display == :verbose
        info("Generating MCL clusters...")
    end
    el2clu, clu_sizes, nunassigned = _mcl_clusters(mcl_adj, allow_singles, tol / length(mcl_adj))

    MCLResult(mcl_adj, el2clu, clu_sizes, nunassigned,
              niter, rel_delta, converged)
end
