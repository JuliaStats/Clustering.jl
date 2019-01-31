# K-means algorithm

####
#### Result object
####

struct KmeansResult{TC<:AbstractFloat,TD<:Real,TCW<:Real} <: ClusteringResult
    centers::Matrix{TC}       # cluster centers (p x k)
    assignments::Vector{Int}  # assignments (n)
    costs::Vector{TD}         # cost of the assignments (n)
    counts::Vector{Int}       # number of samples assigned to each cluster (k)
    cweights::Vector{TCW}     # cluster weights (k)
    totalcost::TD             # total cost (i.e. objective)
    iterations::Int           # number of elapsed iterations
    converged::Bool           # whether the procedure converged
end

const _kmeans_default_init = :kmpp
const _kmeans_default_maxiter = 100
const _kmeans_default_tol = 1.0e-6
const _kmeans_default_display = :none

####
#### Exported kmeans and kmeans! functions
####

function kmeans!(X::AbstractMatrix{TX},                    # in: sample matrix (p x n)
                 centers::AbstractMatrix{TC};              # in: current centers (p x k)
                 weights::Union{Nothing, AbstractVector{TW}}=nothing, # in: sample weights (n)
                 maxiter::Integer=_kmeans_default_maxiter, # in: maximum number of iterations
                 tol::Real=_kmeans_default_tol,            # in: tolerance of change at convergence
                 display::Symbol=_kmeans_default_display,  # in: level of display
                 distance::SemiMetric=SqEuclidean()        # in: function to compute distances
                 ) where {TX<:Real,TC<:Real,TW<:Real}
    p, n = size(X)
    p2, k = size(centers)

    p == p2 || throw(DimensionMismatch("Inconsistent array dimensions " *
                                       "for `X` and `centers`."))
    (2 <= k < n) || error("k must have 2 <= k < n.")
    if !(weights === nothing)
        length(weights) == n || throw(DimensionMismatch("Incorrect length " *
                                                        "of weights."))
    end

    assignments = Vector{Int}(undef, n)
    counts = Vector{Int}(undef, k)

    TC2 = promote_type(TX, TC)
    weights === nothing || (TC2 = promote_type(TC2, TW))
    # corner case where that's still not an AbstractFloat
    TC2 = float(TC2)

    _kmeans!(X, weights, convert(Matrix{TC2}, centers),
             assignments, counts, round(Int, maxiter), tol,
             display_level(display), distance)
end


function kmeans(X::AbstractMatrix{TX}, # in: sample matrix (p x n) columns = observations
                k::Int;                # in: number of centers
                weights::Union{Nothing, AbstractVector{TW}}=nothing, # in: sample weights (n)
                init=_kmeans_default_init,                # in: initialization algorithm
                maxiter::Integer=_kmeans_default_maxiter, # in: maximum number of iterations
                tol::Real=_kmeans_default_tol,            # in: tolerance  of change at convergence
                display::Symbol=_kmeans_default_display,  # in: level of display
                distance::SemiMetric=SqEuclidean()        # in: function to calculate distance with
                ) where {TX<:Real,TW<:Real}
    p, n = size(X)
    (2 <= k < n) || error("k must have 2 <= k < n.")

    TC = float(TX)
    centers = Matrix{TC}(undef, p, k)

    iseeds = initseeds(init, X, k)
    copyseeds!(centers, X, iseeds)

    kmeans!(X, centers;
            weights=weights, maxiter=round(Int, maxiter), tol=tol,
            display=display, distance=distance)
end


####
#### Core implementation
####

function _kmeans!(X::AbstractMatrix{TX},    # in: sample matrix (p x n)
                  weights::Union{Nothing, Vector{TW}}, # in: sample weights (n)
                  centers::Matrix{TC},      # in/out: matrix of centers (p x k)
                  assignments::Vector{Int}, # out: vector of assignments (n)
                  counts::Vector{Int},      # out: # samples assigned to each cluster (k)
                  maxiter::Int,             # in: maximum number of iterations
                  tol::Real,                # in: tolerance of change at convergence
                  displevel::Int,           # in: the level of display
                  distance::SemiMetric      # in: function to calculate the distance with
                  ) where {TX<:Real,TW<:Real,TC<:AbstractFloat}
    p, n = size(X)
    k = size(centers, 2)
    to_update = Vector{Bool}(undef, k) # whether a center needs to be updated
    unused = Vector{Int}()
    num_affected = k # number of centers to which dists need to be recomputed

    # compute pairwise distances, preassign costs and cluster weights
    dmat = pairwise(distance, centers, X)
    costs = Vector{eltype(dmat)}(undef, n)
    TCW = weights === nothing ? Int : TW
    cweights = Vector{TCW}(undef, k)

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

    while !converged && t<maxiter
        t += 1

        # update (affected) centers
        update_centers!(X, weights, assignments, to_update, centers, cweights)

        if !isempty(unused)
            repick_unused_centers(X, costs, centers, unused, distance)
            to_update[unused] .= true
        end

        if t == 1 || num_affected > 0.75 * k
            pairwise!(dmat, distance, centers, X)
        else
            # if only a small subset is affected, only compute for that subset
            affected_inds = findall(to_update)
            pairwise!(view(dmat, affected_inds, :), distance,
                      view(centers, :, affected_inds), X)
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
            @warn("The objective value changes towards an opposite direction")
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

####
#### update assignments
####

function update_assignments!(dmat::Matrix{T},          # in:  distance matrix (k x n)
                             is_init::Bool,            # in:  whether it is the initial run
                             assignments::Vector{Int}, # out: assignment vector (n)
                             costs::Vector{T},         # out: costs of the resultant assignment (n)
                             counts::Vector{Int},      # out: # samples assigned to each cluster (k)
                             to_update::Vector{Bool},  # out: whether a center needs update (k)
                             unused::Vector{Int}       # out: list of centers with no samples
                             ) where T<:Real
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

    # process each sample
    @inbounds for j = 1:n
        # find the closest cluster to the i-th sample
        a = argmin(view(dmat, :, j))
        c = dmat[a, j]

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

    # look for centers that have no associated samples
    for i = 1:k
        if counts[i] == 0
            push!(unused, i)
            to_update[i] = false # this is handled using different mechanism
        end
    end
end

####
#### update centers (unweighted case: weights=nothing)
####

function update_centers!(X::AbstractMatrix{TX},     # in: sample matrix (p x n)
                         weights::Nothing,          # in: sample weights
                         assignments::Vector{Int},  # in: assignments (n)
                         to_update::Vector{Bool},   # in: whether a center needs update (k)
                         centers::Matrix{TC},       # out: updated centers (p x k)
                         cweights::Vector{Int}      # out: updated cluster weights (k)
                         ) where {TX<:Real,TC<:AbstractFloat}
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0

    # accumulate columns
    @inbounds for j ∈ 1:n
        # skip samples assigned to a center that doesn't need to be updated
        cj = assignments[j]
        if to_update[cj]
            if cweights[cj] > 0
                for i ∈ 1:d
                    centers[i, cj] += X[i, j]
                end
            else
                for i ∈ 1:d
                    centers[i, cj] = X[i, j]
                end
            end
            cweights[cj] += 1
        end
    end

    # sum ==> mean
    @inbounds for j ∈ 1:k
        if to_update[j]
            cj = cweights[j]
            for i ∈ 1:d
                centers[i, j] /= cj
            end
        end
    end
end

####
#### update centers (weighted case)
####

function update_centers!(X::AbstractMatrix{TX},     # in: sample matrix (p x n)
                         weights::Vector{TW},       # in: sample weights (n)
                         assignments::Vector{Int},  # in: assignments (n)
                         to_update::Vector{Bool},   # in: whether a center needs update (k)
                         centers::Matrix{TC},       # out: updated centers (p x k)
                         cweights::Vector{TW}       # out: updated cluster weights (k)
                         ) where {TX<:Real,TC<:AbstractFloat,TW<:Real}
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0

    # accumulate columns
    @inbounds for j ∈ 1:n
        # skip samples with negative weights or assigned to a center
        # that doesn't need to be updated
        wj = weights[j]
        cj = assignments[j]
        if wj > 0 && to_update[cj]
            if cweights[cj] > 0
                for i ∈ 1:d
                    centers[i, cj] += X[i, j] * wj
                end
            else
                for i ∈ 1:d
                    centers[i, cj] = X[i, j] * wj
                end
            end
            cweights[cj] += wj
        end
    end

    # sum ==> mean
    @inbounds for j ∈ 1:k
        if to_update[j]
            cj = cweights[j]
            for i ∈ 1:d
                centers[i, j] /= cj
            end
        end
    end
end


####
#### Re-pick centers that get no samples assigned to them.
####

function repick_unused_centers(X::AbstractMatrix{TX},  # in: the sample set (p x n)
                               costs::Vector{TD},      # in: the current assignment costs (n)
                               centers::Matrix{TC},    # to be updated: the centers (p x k)
                               unused::Vector{Int},    # in: set of indices of centers to be updated
                               distance::SemiMetric    # in: function to calculate the distance with
                               ) where {TX<:Real,TC<:AbstractFloat,TD<:Real}
    # pick new centers using a scheme like kmeans++
    ds = similar(costs)
    tcosts = copy(costs)
    n = size(X, 2)

    for i ∈ unused
        j = wsample(1:n, tcosts)
        tcosts[j] = 0
        v = view(X, :, j)
        copyto!(centers[:, i], v)
        colwise!(ds, distance, v, X)
        tcosts = min(tcosts, ds)
    end
end
