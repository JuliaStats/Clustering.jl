# K-medoids algorithm


#### Result type

"""
    KmedoidsResult{T} <: ClusteringResult

The output of [`kmedoids`](@ref) function.

# Fields
 - `medoids::Vector{Int}`: the indices of ``k`` medoids
 - `assignments::Vector{Int}`: the indices of clusters the points are assigned
   to, so that `medoids[assignments[i]]` is the index of the medoid for the
   ``i``-th point
 - `costs::Vector{T}`: assignment costs, i.e. `costs[i]` is the cost of
   assigning ``i``-th point to its medoid
 - `counts::Vector{Int}`: cluster sizes
 - `totalcost::Float64`: total assignment cost (the sum of `costs`)
 - `iterations::Int`: the number of executed algorithm iterations
 - `converged::Bool`: whether the procedure converged
"""
mutable struct KmedoidsResult{T} <: ClusteringResult
    medoids::Vector{Int}        # indices of methods (k)
    assignments::Vector{Int}    # assignments (n)
    costs::Vector{T}            # costs of the resultant assignments (n)
    counts::Vector{Int}         # number of points assigned to each cluster (k)
    totalcost::Float64          # total assignment cost (i.e. objective) (k)
    iterations::Int             # number of elapsed iterations
    converged::Bool             # whether the procedure converged
end

# FIXME remove after deprecation period for acosts
Base.propertynames(kmed::KmedoidsResult, private::Bool = false) =
    (fieldnames(kmed)..., #= deprecated since v0.13.4=# :acosts)

# FIXME remove after deprecation period for acosts
function Base.getproperty(kmed::KmedoidsResult, prop::Symbol)
    if prop == :acosts # deprecated since v0.13.4
        Base.depwarn("KmedoidsResult::acosts is deprecated, use KmedoidsResult::costs", Symbol("KmedoidsResult::costs"))
        return getfield(kmed, :costs)
    else
        return getfield(kmed, prop)
    end
end


#### interface functions

const _kmed_default_init = :kmpp
const _kmed_default_maxiter = 200
const _kmed_default_tol = 1.0e-8
const _kmed_default_display = :none

"""
    kmedoids(dist::DenseMatrix, k::Integer; ...) -> KmedoidsResult

Perform K-medoids clustering of ``n`` points into `k` clusters,
given the `dist` matrix (``n×n``, `dist[i, j]` is the distance
between the `j`-th and `i`-th points).

# Note
This package implements a K-means style algorithm instead of PAM, which
is considered much more efficient and reliable.

# Arguments
 - `init` (defaults to `:kmpp`): how medoids should be initialized, could
   be one of the following:
   * a `Symbol` indicating the name of a seeding algorithm (see
     [Seeding](@ref Seeding) for a list of supported methods).
   * an integer vector of length `k` that provides the indices of points to
     use as initial medoids.
 - `maxiter`, `tol`, `display`: see [common options](@ref common_options)
"""
function kmedoids(dist::DenseMatrix{T}, k::Integer;
                  init=_kmed_default_init,
                  maxiter::Integer=_kmed_default_maxiter,
                  tol::Real=_kmed_default_tol,
                  display::Symbol=_kmed_default_display) where T<:Real
    # check arguments
    n = size(dist, 1)
    size(dist, 2) == n || throw(ArgumentError("dist must be a square matrix ($(size(dist)) given)."))
    k <= n || throw(ArgumentError("Requested number of medoids exceeds n=$n ($k given)."))

    # initialize medoids
    medoids = initseeds_by_costs(init, dist, k)::Vector{Int}
    @assert length(medoids) == k

    # invoke core algorithm
    _kmedoids!(medoids, dist,
               round(Int, maxiter), tol, display_level(display))
end

"""
    kmedoids!(dist::DenseMatrix, medoids::Vector{Int};
              [kwargs...]) -> KmedoidsResult

Update the current cluster `medoids` using the `dist` matrix.

The `medoids` field of the returned `KmedoidsResult` points to the same array
as `medoids` argument.

See [`kmedoids`](@ref) for the description of optional `kwargs`.
"""
function kmedoids!(dist::DenseMatrix{T}, medoids::Vector{Int};
                   maxiter::Integer=_kmed_default_maxiter,
                   tol::Real=_kmed_default_tol,
                   display::Symbol=_kmed_default_display) where T<:Real

    # check arguments
    n = size(dist, 1)
    size(dist, 2) == n ||
        throw(ArgumentError("dist must be a square matrix ($(size(dist)) given)."))
    length(medoids) <= n ||
        throw(ArgumentError("Requested number of medoids exceeds n=$n ($(length(medoids)) given)."))

    # invoke core algorithm
    _kmedoids!(medoids, dist,
               round(Int, maxiter), tol, display_level(display))
end


#### core algorithm

function _kmedoids!(medoids::Vector{Int},      # initialized medoids
                    dist::DenseMatrix{T},      # distance matrix
                    maxiter::Int,              # maximum number of iterations
                    tol::Real,                 # tolerable change of objective
                    displevel::Int) where T<:Real            # level of display

    # dist[i, j] is the cost of assigning point j to the medoid i

    n = size(dist, 1)
    k = length(medoids)

    # prepare storage
    costs = Vector{T}(undef, n)
    counts = zeros(T, k)
    assignments = Vector{Int}(undef, n)

    groups = [Int[] for i=1:k]

    # initialize assignments
    tcost, _ = _kmed_update_assignments!(dist, medoids, assignments, groups, costs, true)

    # main loop
    t = 0
    converged = false

    if displevel >= 2
        @printf("%7s %18s %18s\n", "Iters", "objv", "objv-change")
        println("-----------------------------------------------------")
        @printf("%7d %18.6e\n", t, tcost)
    end

    while !converged && t < maxiter
        t += 1

        # update medoids
        for i = 1:k
            medoids[i] = _find_medoid(dist, groups[i])
        end

        # update assignments
        tcost_pre = tcost
        tcost, ch = _kmed_update_assignments!(dist, medoids, assignments, groups, costs, false)

        # check convergence
        converged = (ch == 0 || abs(tcost - tcost_pre) < tol)

        # display progress
        if displevel >= 2
            @printf("%7d %18.6e %18.6e\n", t, tcost, tcost - tcost_pre)
        end
    end

    if displevel >= 1
        if converged
            println("K-medoids converged with $t iterations (objv = $tcost)")
        else
            println("K-medoids terminated without convergence after $t iterations (objv = $tcost)")
        end
    end

    # make output
    counts = Int[length(g) for g in groups]
    KmedoidsResult{T}(
        medoids,
        assignments,
        costs,
        counts,
        tcost,
        t, converged)
end


# update assignments and related quantities
function _kmed_update_assignments!(dist::DenseMatrix{T},         # in: (n, n)
                                   medoids::AbstractVector{Int}, # in: (k,)
                                   assignments::Vector{Int},     # out: (n,)
                                   groups::Vector{Vector{Int}},  # out: (k,)
                                   costs::Vector{T},             # out: (n,)
                                   isinit::Bool) where T                 # in
    n = size(dist, 1)
    k = length(medoids)
    ch = 0

    if !isinit
        for i = 1:k
            empty!(groups[i])
        end
    end

    tcost = 0.0
    for j = 1:n
        p = 1
        mv = dist[medoids[1], j]

        for i = 2:k
            v = dist[medoids[i], j]
            if v < mv
                p = i
                mv = v
            end
        end

        if isinit
            assignments[j] = p
        else
            a = assignments[j]
            if p != a
                ch += 1
            end
            assignments[j] = p
        end

        costs[j] = mv
        tcost += mv
        push!(groups[p], j)
    end

    return (tcost, ch)::Tuple{Float64, Int}
end


# find medoid for a given group
#
# TODO: faster way without creating temporary arrays
function _find_medoid(dist::DenseMatrix, grp::Vector{Int})
    @assert !isempty(grp)
    p = argmin(sum(dist[grp, grp], dims=2))
    return grp[p]::Int
end
