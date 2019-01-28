# K-means algorithm

#### Interface

mutable struct KmeansResult{T<:Real} <: ClusteringResult
    centers::Matrix{T}         # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{Float64}     # costs of the resultant assignments (n)
    counts::Vector{Int}        # number of samples assigned to each cluster (k)
    cweights::Vector{Float64}  # cluster weights (k)
    totalcost::Float64         # total cost (i.e. objective) (k)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end

const _kmeans_default_init = :kmpp
const _kmeans_default_maxiter = 100
const _kmeans_default_tol = 1.0e-6
const _kmeans_default_display = :none

function kmeans!(X::AbstractMatrix{<:Real}, centers::AbstractMatrix{<:Real};
                 weights=nothing, maxiter::Integer=_kmeans_default_maxiter,
                 tol::Real=_kmeans_default_tol,
                 display::Symbol=_kmeans_default_display,
                 distance::SemiMetric=SqEuclidean())

    m, n = size(X)
    m2, k = size(centers)
    m == m2 || throw(DimensionMismatch("Inconsistent array dimensions."))
    (2 <= k < n) || error("k must have 2 <= k < n.")

    assignments = Vector{Int}(undef, n)
    costs = Vector{Float64}(undef, n)
    counts = Vector{Int}(undef, k)
    cweights = Vector{Float64}(undef, k)

    _kmeans!(X, conv_weights(eltype(X), n, weights), centers, assignments,
             costs, counts, cweights, maxiter, tol,
             display_level(display), distance)
end

function kmeans(X::AbstractMatrix{<:Real}, k::Int;
                weights=nothing, init=_kmeans_default_init,
                maxiter::Integer=_kmeans_default_maxiter,
                tol::Real=_kmeans_default_tol,
                display::Symbol=_kmeans_default_display,
                distance::SemiMetric=SqEuclidean())

    m, n = size(X)
    (2 <= k < n) || error("k must have 2 <= k < n.")
    iseeds = initseeds(init, X, k)
    centers = copyseeds(X, iseeds)
    kmeans!(X, centers;
            weights=weights,
            maxiter=maxiter,
            tol=tol,
            display=display,
            distance=distance)
end

#### Core implementation

# core k-means skeleton
function _kmeans!(
    X::AbstractMatrix{<:Real},                 # in: sample matrix (d x n)
    w::Union{Nothing, AbstractVector{<:Real}}, # in: sample weights (n)
    centers::AbstractMatrix{<:Real},           # in/out: matrix of centers (d x k)
    assignments::Vector{Int},     # out: vector of assignments (n)
    costs::Vector{Float64},       # out: costs of the resultant assignments (n)
    counts::Vector{Int},          # out: # samples assigned to each cluster (k)
    cweights::Vector{Float64},    # out: weights of each cluster
    maxiter::Integer,             # in: maximum number of iterations
    tol::Real,                    # in: tolerance of change at convergence
    displevel::Int,               # in: the level of display
    distance::SemiMetric          # in: function to calculate the distance with
    )

    # initialize
    k = size(centers, 2)
    to_update = Vector{Bool}(undef, k) # indicators of whether a center needs to be updated
    unused = Vector{Int}()
    num_affected::Int = k # number of centers, to which the distances need to be recomputed

    dmat = pairwise(distance, centers, X)
    update_assignments!(dmat, true, assignments, costs, counts, to_update, unused)
    objv = w === nothing ? sum(costs) : dot(w, costs)

    # main loop
    t = 0
    converged = false
    if displevel >= 2
        @printf "%7s %18s %18s | %8s \n" "Iters" "objv" "objv-change" "affected"
        println("-------------------------------------------------------------")
        @printf("%7d %18.6e\n", t, objv)
    end

    while !converged && t < maxiter
        t = t + 1

        # update (affected) centers
        update_centers!(X, w, assignments, to_update, centers, cweights)

        if !isempty(unused)
            repick_unused_centers(X, costs, centers, unused)
        end

        # update pairwise distance matrix
        if !isempty(unused)
            to_update[unused] .= true
        end

        if t == 1 || num_affected > 0.75 * k
            pairwise!(dmat, distance, centers, X)
        else
            # if only a small subset is affected, only compute for that subset
            affected_inds = findall(to_update)
            dmat_p = pairwise(distance, centers[:, affected_inds], X)
            dmat[affected_inds, :] .= dmat_p
        end

        # update assignments

        update_assignments!(dmat, false, assignments, costs, counts, to_update, unused)

        num_affected = sum(to_update) + length(unused)

        # compute change of objective and determine convergence
        prev_objv = objv
        objv = w === nothing ? sum(costs) : dot(w, costs)
        objv_change = objv - prev_objv

        if objv_change > tol
            @warn("The objective value changes towards an opposite direction")
        end

        if abs(objv_change) < tol
            converged = true
        end

        # display iteration information (if asked)

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

    return KmeansResult(convert(Matrix, centers), assignments, costs, counts,
                        cweights, Float64(objv), t, converged)
end


#
#  Updates assignments, costs, and counts based on
#  an updated (squared) distance matrix
#
function update_assignments!(
    dmat::AbstractMatrix{<:Real}, # in:  distance matrix (k x n)
    is_init::Bool,                # in:  whether it is the initial run
    assignments::Vector{Int},     # out: assignment vector (n)
    costs::Vector{Float64},       # out: costs of the resultant assignment (n)
    counts::Vector{Int},          # out: # samples assigned to each cluster (k)
    to_update::Vector{Bool},      # out: whether a center needs update (k)
    unused::Vector{Int})          # out: list of centers with no samples

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
        a = 1
        c = dmat[1, j]
        for i = 2:k
            ci = dmat[i, j]
            if ci < c
                a = i
                c = ci
            end
        end

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

#
#  Update centers based on updated assignments
#
#  (specific to the case where samples are not weighted)
#
function update_centers!(
    X::AbstractMatrix{<:Real},       # in: sample matrix (d x n)
    w::Nothing,                      # in: sample weights
    assignments::Vector{Int},        # in: assignments (n)
    to_update::Vector{Bool},         # in: whether a center needs update (k)
    centers::AbstractMatrix{<:Real}, # out: updated centers (d x k)
    cweights::Vector{Float64})       # out: updated cluster weights (k)

    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0.0

    # accumulate columns
    @inbounds for j = 1:n
        cj = assignments[j]
        1 <= cj <= k || error("assignment out of boundary.")
        if to_update[cj]
            if cweights[cj] > 0
                for i = 1:d
                    centers[i, cj] += X[i, j]
                end
            else
                for i = 1:d
                    centers[i, cj] = X[i, j]
                end
            end
            cweights[cj] += 1
        end
    end

    # sum ==> mean
    @inbounds for j = 1:k
        if to_update[j]
            for i = 1:d
                centers[i, j] /= cweights[j]
            end
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where samples are weighted)
#
function update_centers!(
    X::AbstractMatrix{<:Real},       # in: sample matrix (d x n)
    weights::AbstractVector{<:Real}, # in: sample weights (n)
    assignments::Vector{Int},        # in: assignments (n)
    to_update::Vector{Bool},         # in: whether a center needs update (k)
    centers::AbstractMatrix{<:Real}, # out: updated centers (d x k)
    cweights::Vector{Float64})       # out: updated cluster weights (k)

    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0.0

    # accumulate columns
    @inbounds for j = 1:n
        wj = weights[j]
        if wj > 0
            cj = assignments[j]
            1 <= cj <= k || error("assignment out of boundary.")
            if to_update[cj]
                if cweights[cj] > 0
                    for i = 1:d
                        centers[i, cj] += X[i, j] * wj
                    end
                else
                    for i = 1:d
                        centers[i, cj] = X[i, j] * wj
                    end
                end
                cweights[cj] += wj
            end
        end
    end

    # sum ==> mean
    @inbounds for j = 1:k
        if to_update[j]
            for i = 1:d
                centers[i, j] /= cweights[j]
            end
        end
    end
end


#
#  Re-picks centers that get no samples assigned to them.
#
function repick_unused_centers(
    X::AbstractMatrix{<:Real},       # in: the sample set (d x n)
    costs::Vector{Float64},          # in: the current assignment costs (n)
    centers::AbstractMatrix{<:Real}, # to be updated: the centers (d x k)
    unused::Vector{Int})             # in: set of indices of centers to be updated

    # pick new centers using a scheme like kmeans++
    ds = similar(costs)
    tcosts = copy(costs)
    n = size(X, 2)

    for i in unused
        j = wsample(1:n, tcosts)
        tcosts[j] = 0
        v = view(X, :, j)
        centers[:, i] = v

        colwise!(ds, SqEuclidean(), v, X)
        tcosts = min(tcosts, ds)
    end
end
