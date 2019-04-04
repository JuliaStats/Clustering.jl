# MCL (Markov CLustering algorithm)

"""
    MCLResult <: ClusteringResult

The output of [`mcl`](@ref) function.

# Fields
 - `mcl_adj::AbstractMatrix`: the final MCL adjacency matrix
   (equilibrium state matrix if the algorithm converged), empty if
   `save_final_matrix` option is disabled
 - `assignments::Vector{Int}`: indices of the points clusters.
   `assignments[i]` is the index of the cluster for the ``i``-th point
    (``0`` if unassigned)
 - `counts::Vector{Int}`: the ``k``-length vector of cluster sizes
 - `nunassigned::Int`: the number of standalone points not assigned to any
   cluster
 - `iterations::Int`: the number of elapsed iterations
 - `rel_Δ::Float64`: the final relative Δ
 - `converged::Bool`: whether the method converged
"""
struct MCLResult <: ClusteringResult
    mcl_adj::AbstractMatrix     # final MCL adjacency matrix (equilibrium state matrix if converged)
    assignments::Vector{Int}    # point-to-cluster assignments (n)
    counts::Vector{Int}         # number of points assigned to each cluster (k)
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
    el2clu_mask = mcl_adj[dropdims(sum(mcl_adj, dims=2), dims=2) .> zero_tol, :] .> zero_tol

    # assign cluster indexes to each node
    # cluster index is the index of the first TRUE in a given column
    _ms = mapslices(el_mask->isempty(el_mask) ? 0 : argmax(el_mask), el2clu_mask, dims=1)
    clu_ixs = dropdims(_ms, dims=1)
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
    mcl(adj::AbstractMatrix; [kwargs...]) -> MCLResult

Perform MCL (Markov Cluster Algorithm) clustering using ``n×n``
adjacency (points similarity) matrix `adj`.

# Arguments
Keyword arguments to control the MCL algorithm:
 - `add_loops::Bool` (enabled by default): whether the edges of weight 1.0
   from the node to itself should be appended to the graph
 - `expansion::Number` (defaults to 2): MCL *expansion* constant
 - `inflation::Number` (defaults to 2): MCL *inflation* constant
 - `save_final_matrix::Bool` (disabled by default): whether to save the final
   equilibrium state in the `mcl_adj` field of the result; could provide useful
   diagnostic if the method doesn't converge
 - `prune_tol::Number`: pruning threshold
 - `display`, `maxiter`, `tol`: see [common options](@ref common_options)

# References
> Stijn van Dongen, *"Graph clustering by flow simulation"*, 2001

> [Original MCL implementation](http://micans.org/mcl).
"""
function mcl(adj::AbstractMatrix{T};
             add_loops::Bool = true,
             expansion::Number = 2, inflation::Number = 2,
             save_final_matrix::Bool = false,
             allow_singles::Bool = true,
             max_iter::Union{Integer, Nothing} = nothing,
             maxiter::Integer = 100, tol::Number=1.0e-5,
             prune_tol::Number=1.0e-5, display::Symbol=:none) where T<:Real
    m, n = size(adj)
    m == n || throw(DimensionMismatch("Square adjacency matrix expected"))

    # FIXME max_iter is deprecated as of 0.13.1
    if max_iter !== nothing
        Base.depwarn("max_iter parameter is deprecated, use maxiter instead",
                     Symbol("mcl"))
        maxiter = max_iter
    end

    # FIXME :verbose is deprecated as of 0.13.1
    if display == :verbose
        Base.depwarn("display=:verbose is deprecated and will be removed in future versions, use display=:iter",
                     Symbol("mcl"))
        display = :iter
    end
    disp_level = display_level(display)

    if add_loops
        @inbounds for i in 1:size(adj, 1)
            adj[i, i] = 1.0
        end
    end

    # initialize the MCL adjacency matrix by normalized `adj` weights
    mcl_adj = copy(adj)
    # normalize in columns
    rmul!(mcl_adj, Diagonal(map(x -> x != 0.0 ?  1.0/x : x, dropdims(sum(mcl_adj, dims=1), dims=1))))
    mcl_norm = norm(mcl_adj)
    if !isfinite(mcl_norm)
        throw(OverflowError("The norm of the input adjacency matrix is not finite"))
    end
    next_mcl_adj = similar(mcl_adj)

    # do MCL iterations
    (disp_level > 0) && @info("Starting MCL iterations...")
    niter = 0
    converged = false
    rel_delta = NaN
    while !converged && niter < maxiter
        expanded = _mcl_expand(mcl_adj, expansion)
        _mcl_inflate!(next_mcl_adj, expanded, inflation)
        _mcl_prune!(next_mcl_adj, prune_tol)

        # normalize in columns
        rmul!(next_mcl_adj, Diagonal(map(x -> x != 0.0 ? 1.0/x : x,
                                         dropdims(sum(next_mcl_adj, dims=1), dims=1))))

        next_mcl_norm = norm(next_mcl_adj)
        if !isfinite(next_mcl_norm)
            @warn("MCL adjacency matrix norm is not finite")
            break
        end
        rel_delta = euclidean(next_mcl_adj, mcl_adj)/mcl_norm
        (disp_level == 2) && @info("MCL iter. #$niter: rel.Δ=", rel_delta)
        (converged = rel_delta <= tol) && break
        # update (swap) MCL adjacency
        niter += 1
        mcl_adj, next_mcl_adj = next_mcl_adj, mcl_adj
        mcl_norm = next_mcl_norm
        (mcl_norm < tol) && break # matrix is zero
    end

    if disp_level > 0
        if converged
            @info "MCL converged after $niter iteration(s)"
        else
            @warn "MCL didn't converge after $niter iteration(s)"
        end
    end

    (disp_level > 0) && @info("Generating MCL clusters...")
    el2clu, clu_sizes, nunassigned = _mcl_clusters(mcl_adj, allow_singles,
                                                   tol/length(mcl_adj))

    return MCLResult(save_final_matrix ? mcl_adj : similar(mcl_adj, (0,0)),
              el2clu, clu_sizes, nunassigned, niter, rel_delta, converged)
end
