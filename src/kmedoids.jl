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

#### interface functions

const _kmed_default_init = :kmpp
const _kmed_default_maxiter = 200
const _kmed_default_tol = 1.0e-8
const _kmed_default_display = :none

"""
    kmedoids(dist::AbstractMatrix, k::Integer; ...) -> KmedoidsResult

Perform K-medoids clustering of ``n`` points into `k` clusters,
given the `dist` matrix (``n×n``, `dist[i, j]` is the distance
between the `j`-th and `i`-th points).

# Arguments
 - `init` (defaults to `:kmpp`): how medoids should be initialized, could
   be one of the following:
   * a `Symbol` indicating the name of a seeding algorithm (see
     [Seeding](@ref Seeding) for a list of supported methods).
   * an integer vector of length `k` that provides the indices of points to
     use as initial medoids.
 - `maxiter`, `tol`, `display`: see [common options](@ref common_options)

# Note
The function implements a *K-means style* algorithm instead of *PAM*
(Partitioning Around Medoids). K-means style algorithm converges in fewer
iterations, but was shown to produce worse (10-20% higher total costs) results
(see e.g. [Schubert & Rousseeuw (2019)](@ref kmedoid_refs)).
"""
function kmedoids(dist::AbstractMatrix{T}, k::Integer;
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
    kmedoids!(dist::AbstractMatrix, medoids::Vector{Int};
              [kwargs...]) -> KmedoidsResult

Update the current cluster `medoids` using the `dist` matrix.

The `medoids` field of the returned `KmedoidsResult` points to the same array
as `medoids` argument.

See [`kmedoids`](@ref) for the description of optional `kwargs`.
"""
function kmedoids!(dist::AbstractMatrix{T}, medoids::Vector{Int};
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
                    dist::AbstractMatrix{T},   # distance matrix
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
# returns the total cost and the number of assignment changes
function _kmed_update_assignments!(dist::AbstractMatrix{<:Real}, # in: (n, n)
                                   medoids::AbstractVector{Int}, # in: (k,)
                                   assignments::Vector{Int},     # out: (n,)
                                   groups::Vector{Vector{Int}},  # out: (k,)
                                   costs::AbstractVector{<:Real},# out: (n,)
                                   initial::Bool)                # in
    n = size(dist, 1)
    k = length(medoids)

    # reset cluster groups (note: assignments are not touched yet)
    initial || foreach(empty!, groups)

    tcost = 0.0
    ch = 0
    for j = 1:n
        p = 1 # initialize the closest medoid for j
        mv = dist[medoids[1], j]

        # find the closest medoid for j
        @inbounds for i = 2:k
            m = medoids[i]
            v = dist[m, j]
            # assign if current medoid is closer or if it is j itself
            if (v < mv) || (m == j)
                (v <= mv) || throw(ArgumentError("sample #$j reassigned from medoid[$p]=#$(medoids[p]) (distance=$mv) to medoid[$i]=#$m (distance=$v); check the distance matrix correctness"))
                p = i
                mv = v
            end
        end

        ch += !initial && (p != assignments[j])
        assignments[j] = p
        costs[j] = mv
        tcost += mv
        push!(groups[p], j)
    end

    return (tcost, ch)
end


# find medoid for a given group
function _find_medoid(dist::AbstractMatrix, grp::AbstractVector{Int})
    @assert !isempty(grp)
    p = argmin(sum(view(dist, grp, grp), dims=2))
    return grp[p]
end
