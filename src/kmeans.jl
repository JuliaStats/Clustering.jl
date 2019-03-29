# K-means algorithm

#### Interface

# C is the type of centers, an (abstract) matrix of size (d x k)
# D is the type of pairwise distance computation from points to cluster centers
# WC is the type of cluster weights, either Int (in the case where points are
# unweighted) or eltype(weights) (in the case where points are weighted).
"""
The output of K-means algorithm.

See also: [`kmeans`](@ref), [`kmeans!`](@ref).
"""
struct KmeansResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real,WC<:Real} <: ClusteringResult
    centers::C                 # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of points assigned to each cluster (k)
    cweights::Vector{WC}       # cluster weights (k)
    totalcost::D               # total cost (i.e. objective)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end

const _kmeans_default_init = :kmpp
const _kmeans_default_maxiter = 100
const _kmeans_default_tol = 1.0e-6
const _kmeans_default_display = :none

"""
    kmeans!(X, centers; [kwargs...])

Update the current cluster `centers` (``d×k`` matrix, where ``d`` is the
dimension and ``k`` the number of centroids) using the ``d×n`` data
matrix `X` (each column of `X` is a ``d``-dimensional data point).

Returns `KmeansResult` object.

See [`kmeans`](@ref) for the description of optional `kwargs`.
"""
function kmeans!(X::AbstractMatrix{<:Real},                # in: data matrix (d x n)
                 centers::AbstractMatrix{<:AbstractFloat}; # in: current centers (d x k)
                 weights::Union{Nothing, AbstractVector{<:Real}}=nothing, # in: data point weights (n)
                 maxiter::Integer=_kmeans_default_maxiter, # in: maximum number of iterations
                 tol::Real=_kmeans_default_tol,            # in: tolerance of change at convergence
                 display::Symbol=_kmeans_default_display,  # in: level of display
                 distance::SemiMetric=SqEuclidean())       # in: function to compute distances
    d, n = size(X)
    dc, k = size(centers)

    d == dc || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (2 <= k < n) || error("k must have 2 <= k < n.")
    if weights !== nothing
        length(weights) == n || throw(DimensionMismatch("Incorrect length of weights."))
    end

    _kmeans!(X, weights, centers, Int(maxiter), Float64(tol),
             display_level(display), distance)
end


"""
    kmeans(X, k, [...])

K-means clustering of the ``d×n`` data matrix `X` (each column of `X`
is a ``d``-dimensional data point) into `k` clusters.

Returns `KmeansResult` object.

# Algorithm Options
 - `init` (defaults to `:kmpp`): how cluster seeds should be initialized, could
   be one of the following:
   * a `Symbol`, the name of a seeding algorithm (see [Seeding](@ref) for a list
     of supported methods).
   * an integer vector of length ``k`` that provides the indices of points to
     use as initial seeds.
 - `weights`: ``n``-element vector of point weights (the cluster centers are
   the weighted means of cluster members)
 - `maxiter`, `tol`, `display`: see [common options](@ref common_options)
"""
function kmeans(X::AbstractMatrix{<:Real},                # in: data matrix (d x n) columns = obs
                k::Integer;                               # in: number of centers
                weights::Union{Nothing, AbstractVector{<:Real}}=nothing, # in: data point weights (n)
                init::Symbol=_kmeans_default_init,        # in: initialization algorithm
                maxiter::Integer=_kmeans_default_maxiter, # in: maximum number of iterations
                tol::Real=_kmeans_default_tol,            # in: tolerance  of change at convergence
                display::Symbol=_kmeans_default_display,  # in: level of display
                distance::SemiMetric=SqEuclidean())       # in: function to calculate distance with
    d, n = size(X)
    (2 <= k < n) || throw(ArgumentError("k must be 2 <= k < n, k=$k given."))

    # initialize the centers using a type wide enough so that the updates
    # centers[i, cj] += X[i, j] * wj will occur without loss of precision through rounding
    T = float(weights === nothing ? eltype(X) : promote_type(eltype(X), eltype(weights)))
    iseeds = initseeds(init, X, k)
    centers = copyseeds!(Matrix{T}(undef, d, k), X, iseeds)

    kmeans!(X, centers;
            weights=weights, maxiter=Int(maxiter), tol=Float64(tol),
            display=display, distance=distance)
end

#### Core implementation

# core k-means skeleton
function _kmeans!(X::AbstractMatrix{<:Real},                # in: data matrix (d x n)
                  weights::Union{Nothing, Vector{<:Real}},  # in: data point weights (n)
                  centers::AbstractMatrix{<:AbstractFloat}, # in/out: matrix of centers (d x k)
                  maxiter::Int,                             # in: maximum number of iterations
                  tol::Float64,                             # in: tolerance of change at convergence
                  displevel::Int,                           # in: the level of display
                  distance::SemiMetric)                     # in: function to calculate distance
    d, n = size(X)
    k = size(centers, 2)
    to_update = Vector{Bool}(undef, k) # whether a center needs to be updated
    unused = Vector{Int}()
    num_affected = k # number of centers to which dists need to be recomputed

    # assign containers for the vector of assignments & number of data points assigned to each cluster
    assignments = Vector{Int}(undef, n)
    counts = Vector{Int}(undef, k)

    # compute pairwise distances, preassign costs and cluster weights
    dmat = pairwise(distance, centers, X, dims=2)
    WC = (weights === nothing) ? Int : eltype(weights)
    cweights = Vector{WC}(undef, k)
    D = typeof(one(eltype(dmat)) * one(WC))
    costs = Vector{D}(undef, n)

    update_assignments!(dmat, true, assignments, costs, counts,
                        to_update, unused)
    objv = weights === nothing ? sum(costs) : dot(weights, costs)

    # main loop
    t = 0
    converged = false
    if displevel >= 2
        @printf "%7s %18s %18s | %8s \n" "Iters" "objv" "objv-change" "affected"
        println("-------------------------------------------------------------")
        @printf("%7d %18.6e\n", t, objv)
    end

    while !converged && t < maxiter
        t += 1

        # update (affected) centers
        update_centers!(X, weights, assignments, to_update, centers, cweights)

        if !isempty(unused)
            repick_unused_centers(X, costs, centers, unused, distance)
            to_update[unused] .= true
        end

        if t == 1 || num_affected > 0.75 * k
            pairwise!(dmat, distance, centers, X, dims=2)
        else
            # if only a small subset is affected, only compute for that subset
            affected_inds = findall(to_update)
            pairwise!(view(dmat, affected_inds, :), distance,
                      view(centers, :, affected_inds), X, dims=2)
        end

        # update assignments
        update_assignments!(dmat, false, assignments, costs, counts,
                            to_update, unused)
        num_affected = sum(to_update) + length(unused)

        # compute change of objective and determine convergence
        prev_objv = objv
        objv = weights === nothing ? sum(costs) : dot(weights, costs)
        objv_change = objv - prev_objv

        if objv_change > tol
            @warn("The clustering cost increased at iteration #$t")
        elseif abs(objv_change) < tol
            converged = true
        end

        # display information (if required)
        if displevel >= 2
            @printf("%7d %18.6e %18.6e | %8d\n", t, objv, objv_change, num_affected)
        end
    end

    if displevel >= 1
        if converged
            println("K-means converged with $t iterations (objv = $objv)")
        else
            println("K-means terminated without convergence after $t iterations (objv = $objv)")
        end
    end

    return KmeansResult(centers, assignments, costs, counts,
                        cweights, objv, t, converged)
end

#
#  Updates assignments, costs, and counts based on
#  an updated (squared) distance matrix
#
function update_assignments!(dmat::Matrix{<:Real},     # in:  distance matrix (k x n)
                             is_init::Bool,            # in:  whether it is the initial run
                             assignments::Vector{Int}, # out: assignment vector (n)
                             costs::Vector{<:Real},    # out: costs of the resultant assignment (n)
                             counts::Vector{Int},      # out: # of points assigned to each cluster (k)
                             to_update::Vector{Bool},  # out: whether a center needs update (k)
                             unused::Vector{Int}       # out: list of centers with no points assigned
                             )
    k, n = size(dmat)

    # re-initialize the counting vector
    fill!(counts, 0)

    if is_init
        fill!(to_update, true)
    else
        fill!(to_update, false)
        if !isempty(unused)
            empty!(unused)
        end
    end

    # process each point
    @inbounds for j = 1:n
        # find the closest cluster to the i-th point. Note that a
        # is necessarily between 1 and size(dmat, 1) === k as a result
        # and can thus be used as an index in an `inbounds` environment
        c, a = findmin(view(dmat, :, j))

        # set/update the assignment
        if is_init
            assignments[j] = a
        else  # update
            pa = assignments[j]
            if pa != a
                # if assignment changes,
                # both old and new centers need to be updated
                assignments[j] = a
                to_update[a] = true
                to_update[pa] = true
            end
        end

        # set costs and counts accordingly
        costs[j] = c
        counts[a] += 1
    end

    # look for centers that have no assigned points
    for i = 1:k
        if counts[i] == 0
            push!(unused, i)
            to_update[i] = false # this is handled using different mechanism
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where points are not weighted)
#
function update_centers!(X::AbstractMatrix{<:Real},        # in: data matrix (d x n)
                         weights::Nothing,                 # in: point weights
                         assignments::Vector{Int},         # in: assignments (n)
                         to_update::Vector{Bool},          # in: whether a center needs update (k)
                         centers::AbstractMatrix{<:AbstractFloat}, # out: updated centers (d x k)
                         cweights::Vector{Int})            # out: updated cluster weights (k)
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0

    # accumulate columns
    @inbounds for j in 1:n
        # skip points assigned to a center that doesn't need to be updated
        cj = assignments[j]
        if to_update[cj]
            if cweights[cj] > 0
                for i in 1:d
                    centers[i, cj] += X[i, j]
                end
            else
                for i in 1:d
                    centers[i, cj] = X[i, j]
                end
            end
            cweights[cj] += 1
        end
    end

    # sum ==> mean
    @inbounds for j in 1:k
        if to_update[j]
            cj = cweights[j]
            for i in 1:d
                centers[i, j] /= cj
            end
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where points are weighted)
#
function update_centers!(X::AbstractMatrix{<:Real}, # in: data matrix (d x n)
                         weights::Vector{W},        # in: point weights (n)
                         assignments::Vector{Int},  # in: assignments (n)
                         to_update::Vector{Bool},   # in: whether a center needs update (k)
                         centers::AbstractMatrix{<:Real}, # out: updated centers (d x k)
                         cweights::Vector{W}        # out: updated cluster weights (k)
                         ) where W<:Real
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0

    # accumulate columns
    @inbounds for j in 1:n
        # skip points with negative weights or assigned to a center
        # that doesn't need to be updated
        wj = weights[j]
        cj = assignments[j]
        if wj > 0 && to_update[cj]
            if cweights[cj] > 0
                for i in 1:d
                    centers[i, cj] += X[i, j] * wj
                end
            else
                for i in 1:d
                    centers[i, cj] = X[i, j] * wj
                end
            end
            cweights[cj] += wj
        end
    end

    # sum ==> mean
    @inbounds for j in 1:k
        if to_update[j]
            cj = cweights[j]
            for i in 1:d
                centers[i, j] /= cj
            end
        end
    end
end


#
#  Re-picks centers that have no points assigned to them.
#
function repick_unused_centers(X::AbstractMatrix{<:Real}, # in: the data matrix (d x n)
                               costs::Vector{<:Real},     # in: the current assignment costs (n)
                               centers::AbstractMatrix{<:AbstractFloat}, # out: the centers (d x k)
                               unused::Vector{Int},       # in: indices of centers to be updated
                               distance::SemiMetric)      # in: function to calculate the distance with
    # pick new centers using a scheme like kmeans++
    ds = similar(costs)
    tcosts = copy(costs)
    n = size(X, 2)

    for i in unused
        j = wsample(1:n, tcosts)
        tcosts[j] = 0
        v = view(X, :, j)
        centers[:, i] = v
        colwise!(ds, distance, v, X)
        tcosts = min(tcosts, ds)
    end
end
