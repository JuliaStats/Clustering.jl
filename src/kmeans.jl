# K-means algorithm

####
#### Result object
####

# NOTE REFACT: there's no need for this to be a mutable struct.

struct KmeansResult{T<:AbstractFloat} <: ClusteringResult
    centers::Matrix{T}        # cluster centers (d x k)
    assignments::Vector{Int}  # assignments (n)
    costs::Vector{T}          # cost of the assignments (n)
    counts::Vector{Int}       # number of samples assigned to each cluster (k)
    cweights::Vector{T}       # cluster weights (k)
    totalcost::T              # total cost (i.e. objective)
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

# Case when TX is a Real but not a AbstractFloat
function kmeans!(X::AbstractMatrix{<:Union{Integer, Rational, AbstractIrrational}},
                 centers::AbstractMatrix{<:Real}; args...)
    # NOTE REFACT: a warning could be issued here
    kmeans!(convert(Matrix{Float64}, X), centers; args...)
end

# case when TX<:AbstractFloat
function kmeans!(X::AbstractMatrix{TX}, centers::AbstractMatrix{TC};
                 weights::Union{Nothing, AbstractVector{<:Real}}=nothing,
                 maxiter::Int=_kmeans_default_maxiter,
                 tol::Real=_kmeans_default_tol,
                 display::Symbol=_kmeans_default_display,
                 distance::SemiMetric=SqEuclidean()
                 ) where {TX<:AbstractFloat,TC<:Real}
    p, n = size(X)
    p2, k = size(centers)

    p == p2 || throw(DimensionMismatch("Inconsistent array dimensions " *
                                       "for `X` and `centers`."))
    (2 <= k < n) || error("k must have 2 <= k < n.")

    assignments = Vector{Int}(undef, n)
    costs = Vector{TX}(undef, n) # distance to centroid, will be of type TX.
    counts = Vector{Int}(undef, k)
    cweights = Vector{TX}(undef, k)

    # NOTE REFACT: since the weights here are converted to TX, we're sure that
    # (a) it's still floating point if it needs to be (b) X[:, j] * w[j]
    # preserves the type TX. The only case where there would be a possible
    # loss of information is if TX is simpler than TW so for instance Float32
    # versus Float64. In that case the choice is to keep things in Float32.
    # NOTE REFACT: The centers can be provided with any type but since they
    # will be updated with updates of type TX it makes sense to convert them
    # to TX as well if they aren't already.
    # NOTE REFACT: so at this point, the passed X, weights and centers all
    # have the same type TX which will be preserved.
    _kmeans!(X, conv_weights(TX, n, weights), convert(Matrix{TX}, centers),
             assignments, costs, counts, cweights, maxiter, tol,
             display_level(display), distance)
end

# Case when TX is a Real but not a AbstractFloat
function kmeans(X::AbstractMatrix{<:Union{Integer, Rational, AbstractIrrational}}, k::Int; args...)
    # NOTE REFACT: a warning could be issued here
    kmeans(convert(Matrix{Float64}, X), k; args...)
end

# Case when TX is a AbstractFloat
function kmeans(X::AbstractMatrix{TX}, k::Int;
                weights::Union{Nothing, AbstractVector{TW}}=nothing,
                init=_kmeans_default_init,
                maxiter::Int=_kmeans_default_maxiter,
                tol::Real=_kmeans_default_tol,
                display::Symbol=_kmeans_default_display,
                distance::SemiMetric=SqEuclidean()
                ) where {TX<:AbstractFloat,TW<:Real}
    p, n = size(X)
    (2 <= k < n) || error("k must have 2 <= k < n.")

    iseeds = initseeds(init, X, k)
    centers = copyseeds(X, iseeds)

    kmeans!(X, centers;
            weights=weights, maxiter=maxiter, tol=tol,
            display=display, distance=distance)
end


####
#### Core implementation
####

# NOTE REFACT: this has been called *after* conversions and so necessarily
# the eltype(X) == eltype(weights) (if any) == eltype(centers). Also
# centers isa Matrix and weights isa Vector (or nothing)
function _kmeans!(X::AbstractMatrix{T},
                  weights::Union{Nothing, Vector{T}},
                  centers::Matrix{T},
                  assignments::Vector{Int},
                  costs::Vector{T},
                  counts::Vector{Int},
                  cweights::Vector{T},
                  maxiter::Int,
                  tol::Real,
                  displevel::Int,
                  distance::SemiMetric
                  ) where {T<:AbstractFloat}
    k = size(centers, 2)
    to_update = Vector{Bool}(undef, k) # whether a center needs to be updated
    unused = Vector{Int}()
    num_affected = k # # centers to which distances need to be recomputed

    # compute pairwise distances
    dmat = pairwise(distance, centers, X)
    dmat = convert(Matrix{T}, dmat)

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
            @printf("%7d %18.6e %18.6e | %8d\n", t, objv, objv_change,
                    num_affected)
        end
    end # end while

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

function update_assignments!(dmat::Matrix{T}, is_init::Bool,
                             assignments::Vector{Int}, costs::Vector{T},
                             counts::Vector{Int}, to_update::Vector{Bool},
                             unused::Vector{Int}
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

function update_centers!(X::AbstractMatrix{T}, ::Nothing,
                         assignments::Vector{Int}, to_update::Vector{Bool},
                         centers::Matrix{T}, cweights::Vector{T}
                         ) where T<:AbstractFloat
    # NOTE REFACT: does this ever happen?
    # The check that assignments are between 1 and k is pointless since
    # the only element that goes in an assignment is the result of
    # a = argmin(view(dmat, :, j)) which is necessarily between 1 and k
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= zero(T)
    oneT = one(T)

    # accumulate columns
    @inbounds for j ∈ 1:n
        # skip samples assigned to a center that doesn't need to be updated
        cj = assignments[j]
        if to_update[cj]
            if cweights[cj] > 0
                for i ∈ 1:d
                    # NOTE REFACT: everything is type T here
                    centers[i, cj] += X[i, j]
                end
            else
                for i ∈ 1:d
                    centers[i, cj] = X[i, j]
                end
            end
            # NOTE REFACT: everything is of type T here
            cweights[cj] += oneT
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

function update_centers!(X::AbstractMatrix{T}, weights::Vector{T},
                         assignments::Vector{Int}, to_update::Vector{Bool},
                         centers::Matrix{T}, cweights::Vector{T}
                         ) where T<:AbstractFloat
    d, n = size(X)
    k = size(centers, 2)

    # initialize center weights
    cweights[to_update] .= zero(T)

    # accumulate columns
    @inbounds for j ∈ 1:n
        # skip samples with negative weights or assigned to a center
        # that doesn't need to be updated
        wj = weights[j]
        cj = assignments[j]
        if to_update[cj] && wj > 0
            if cweights[cj] > 0
                for i ∈ 1:d
                    # NOTE REFACT: everything is type T here
                    centers[i, cj] += X[i, j] * wj
                end
            else
                for i ∈ 1:d
                    centers[i, cj] = X[i, j] * wj
                end
            end
            # NOTE REFACT: everything is type T here
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

function repick_unused_centers(X::AbstractMatrix{T}, costs::Vector{T},
                               centers::Matrix{T}, unused::Vector{Int},
                               distance::SemiMetric
                               ) where T<:AbstractFloat
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
