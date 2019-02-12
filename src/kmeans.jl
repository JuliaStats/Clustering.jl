# K-means algorithm

####
#### Result object
####

# T is the eltype(centers)
# D is the type of pairwise distance computation from samples to centers
# WC is the type of cluster weights, either Int (in the case where samples are
# unweighted) or eltype(weights) (in the case where samples are weighted).
struct KmeansResult{T<:AbstractFloat,D<:Real,WC<:Real} <: ClusteringResult
    centers::AbstractMatrix{T} # cluster centers (p x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of samples assigned to each cluster (k)
    cweights::Vector{WC}       # cluster weights (k)
    totalcost::D               # total cost (i.e. objective)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end

const _kmeans_default_init = :kmpp
const _kmeans_default_maxiter = 100
const _kmeans_default_tol = 1.0e-6
const _kmeans_default_display = :none

####
#### Exported kmeans and kmeans! functions
####

"""
    kmeans!(centers, X)

Update the current centers `centers` (of size `p x k` where `p` is the dimension and `k` the
number of centroids) using the samples contained in `X` (of size `p x n` where `n` is the number
of samples).
"""
function kmeans!(centers::AbstractMatrix{<:AbstractFloat}, # in: current centers (p x k)
                 X::AbstractMatrix{<:Real};                # in: sample matrix (p x n)
                 weights::Union{Nothing, AbstractVector{<:Real}}=nothing, # in: sample weights (n)
                 maxiter::Integer=_kmeans_default_maxiter, # in: maximum number of iterations
                 tol::Real=_kmeans_default_tol,            # in: tolerance of change at convergence
                 display::Symbol=_kmeans_default_display,  # in: level of display
                 distance::SemiMetric=SqEuclidean())       # in: function to compute distances
    p, n = size(X)
    p2, k = size(centers)

    p == p2 || throw(DimensionMismatch("Inconsistent array dimensions for `X` and `centers`."))
    (2 <= k < n) || error("k must have 2 <= k < n.")
    if weights !== nothing
        length(weights) == n || throw(DimensionMismatch("Incorrect length of weights."))
    end

    assignments = Vector{Int}(undef, n)
    counts = Vector{Int}(undef, k)

    # check the types to see wheter the updates: centers[i, cj] += X[i, j] * wj
    # may occur loss of precision through silent casting
    update_type = float(weights === nothing ? eltype(X) : typeof(one(eltype(X)) * one(eltype(weights))))
    update_type <: eltype(centers) || @warn "The type of the centers update ($update_type) is " *
                                            "wider than that of the given centers " *
                                            "($(eltype(centers))). This may incur rounding errors."

    _kmeans!(X, weights, centers, assignments, counts,
             Int(maxiter), Float64(tol), display_level(display), distance)
end


"""
    kmeans(X, k)

K-means clustering with `k` centroids of the data contained in `X` of size `p x n` where `p` is
the dimension and `n` is the number of samples.
"""
function kmeans(X::AbstractMatrix{<:Real},                # in: sample matrix (p x n) columns = obs
                k::Integer;                               # in: number of centers
                weights::Union{Nothing, AbstractVector{<:Real}}=nothing, # in: sample weights (n)
                init::Symbol=_kmeans_default_init,        # in: initialization algorithm
                maxiter::Int=_kmeans_default_maxiter,     # in: maximum number of iterations
                tol::Float64=_kmeans_default_tol,         # in: tolerance  of change at convergence
                display::Symbol=_kmeans_default_display,  # in: level of display
                distance::SemiMetric=SqEuclidean())       # in: function to calculate distance with
    p, n = size(X)
    k = round(Int, k)
    (2 <= k < n) || error("k must have 2 <= k < n.")

    # initialize the centers using a type wide enough so that the updates
    # centers[i, cj] += X[i, j] * wj will occur without loss of precision through rounding
    T = float(weights === nothing ? eltype(X) : promote_type(eltype(X), eltype(weights)))
    iseeds = initseeds(init, X, k)
    centers = copyseeds!(Matrix{T}(undef, p, k), X, iseeds)

    kmeans!(centers, X;
            weights=weights, maxiter=round(Int, maxiter), tol=tol,
            display=display, distance=distance)
end


####
#### Core implementation
####

function _kmeans!(X::AbstractMatrix{<:Real},                # in: sample matrix (p x n)
                  weights::Union{Nothing, Vector{<:Real}},  # in: sample weights (n)
                  centers::AbstractMatrix{<:AbstractFloat}, # in/out: matrix of centers (p x k)
                  assignments::Vector{Int},                 # out: vector of assignments (n)
                  counts::Vector{Int},                      # out: number of samples assigned to each cluster (k)
                  maxiter::Int,                             # in: maximum number of iterations
                  tol::Real,                                # in: tolerance of change at convergence
                  displevel::Int,                           # in: the level of display
                  distance::SemiMetric)                     # in: function to calculate the distance with
    p, n = size(X)
    k = size(centers, 2)
    to_update = Vector{Bool}(undef, k) # whether a center needs to be updated
    unused = Vector{Int}()
    num_affected = k # number of centers to which dists need to be recomputed

    # compute pairwise distances, preassign costs and cluster weights
    dmat = pairwise(distance, centers, X)
    costs = Vector{eltype(dmat)}(undef, n)
    WC = (weights === nothing) ? Int : eltype(weights)
    cweights = Vector{WC}(undef, k)

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
        # find the closest cluster to the i-th sample. Note that a
        # is necessarily between 1 and size(dmat, 1) === k as a result
        # and can thus be used as an index in an `inbounds` environment
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
#### update centers (unweighted case)
####

function update_centers!(X::AbstractMatrix{<:Real},        # in: sample matrix (p x n)
                         weights::Nothing,                 # in: sample weights
                         assignments::Vector{Int},         # in: assignments (n)
                         to_update::Vector{Bool},          # in: whether a center needs update (k)
                         centers::AbstractMatrix{<:AbstractFloat}, # out: updated centers (p x k)
                         cweights::Vector{Int})            # out: updated cluster weights (k)
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0

    # accumulate columns
    @inbounds for j in 1:n
        # skip samples assigned to a center that doesn't need to be updated
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

####
#### update centers (weighted case)
####

function update_centers!(X::AbstractMatrix{<:Real}, # in: sample matrix (p x n)
                         weights::Vector{W},        # in: sample weights (n)
                         assignments::Vector{Int},  # in: assignments (n)
                         to_update::Vector{Bool},   # in: whether a center needs update (k)
                         centers::AbstractMatrix{<:Real}, # out: updated centers (p x k)
                         cweights::Vector{W}        # out: updated cluster weights (k)
                         ) where W<:Real
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= 0

    # accumulate columns
    @inbounds for j in 1:n
        # skip samples with negative weights or assigned to a center
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


####
#### Re-pick centers that get no samples assigned to them.
####

function repick_unused_centers(X::AbstractMatrix{<:Real},  # in: the sample set (p x n)
                               costs::Vector{<:Real},      # in: the current assignment costs (n)
                               centers::AbstractMatrix{<:AbstractFloat}, # out: the centers (p x k)
                               unused::Vector{Int},  # in: set of indices of centers to be updated
                               distance::SemiMetric) # in: function to calculate the distance with
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
