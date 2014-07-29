
###########################################################
#
#   K-means options
#
###########################################################

type KmeansOpts
    max_iters::Int
    tol::Float64
    weights::Union(Nothing, Vector)
    display::Symbol
end


function kmeans_opts(;max_iter::Integer=200, tol::Real=1.0e-6, weights=nothing, display::Symbol=:iter)
    KmeansOpts(
        int(max_iter), 
        float64(tol), 
        weights, 
        display)
end


###########################################################
#
#   Core implementation
#
#   Notations:
#   - d:    the dimension of each sample
#   - n:    the number of samples
#   - k:    the number of clusters
#
###########################################################

#
#  Updates assignments, costs, and counts based on
#  an updated (squared) distance matrix
#
function update_assignments!{T<:FloatingPoint}(
    dmat::Matrix{T},            # in:  distance matrix (k x n)
    is_init::Bool,              # in:  whether it is the initial run
    assignments::Vector{Int},   # out: assignment vector (n)
    costs::Vector{T},           # out: costs of the resultant assignment (n)
    counts::Vector{Int},        # out: number of samples assigned to each cluster (k)
    to_update::Vector{Bool},    # out: whether a center needs update (k)
    unused::Vector{Int})        # out: the list of centers get no samples assigned to it

    k::Int, n::Int = size(dmat)

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
    for j = 1 : n

        # find the closest cluster to the i-th sample
        a::Int = 1
        c::T = dmat[1, j]
        for i = 2 : k
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

    for i = 1 : k
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
function update_centers!{T<:FloatingPoint}(
    x::Matrix{T},                   # in: sample matrix (d x n)
    w::Nothing,                     # in: sample weights
    assignments::Vector{Int},       # in: assignments (n)
    to_update::Vector{Bool},        # in: whether a center needs update (k)
    centers::Matrix{T},             # out: updated centers (d x k)
    cweights::Vector{T})            # out: updated cluster weights (k)

    d::Int = size(x, 1)
    n::Int = size(x, 2)
    k::Int = size(centers, 2)

    # initialize center weights 
    for i = 1 : k
        if to_update[i]
            cweights[i] = 0.
        end
    end

    # accumulate columns
    accumulate_cols_u!(centers, cweights, x, assignments, to_update)

    # sum ==> mean
    for j = 1:k
        if to_update[j]
            @inbounds cj::T = 1 / cweights[j]
            vj = view(centers,:,j)
            for i = 1:d
                @inbounds vj[i] *= cj
            end
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where samples are weighted)
#
function update_centers!{T<:FloatingPoint}(
    x::Matrix{T},                   # in: sample matrix (d x n)
    weights::Vector{T},             # in: sample weights (n)
    assignments::Vector{Int},       # in: assignments (n)
    to_update::Vector{Bool},        # in: whether a center needs update (k)
    centers::Matrix{T},             # out: updated centers (d x k)
    cweights::Vector{T})            # out: updated cluster weights (k)

    d::Int = size(x, 1)
    n::Int = size(x, 2)
    k::Int = size(centers, 2)

    # initialize center weights 
    for i = 1 : k
        if to_update[i]
            cweights[i] = 0.
        end
    end

    # accumulate columns
    accumulate_cols_u!(centers, cweights, x, assignments, weights, to_update)

    # sum ==> mean
    for j = 1:k
        if to_update[j]
            @inbounds cj::T = 1 / cweights[j]
            vj = view(centers,:,j)
            for i = 1:d
                @inbounds vj[i] *= cj
            end
        end
    end
end


#
#  Re-picks centers that get no samples assigned to them.
#
function repick_unused_centers{T<:FloatingPoint}(
    x::Matrix{T},           # in: the sample set (d x n)
    costs::Vector{T},       # in: the current assignment costs (n)
    centers::Matrix{T},     # to be updated: the centers (d x k)
    unused::Vector{Int})    # in: the set of indices of centers to be updated

    # pick new centers using a scheme like kmeans++
    ds = similar(costs)
    tcosts = copy(costs)
    n = size(x, 2)

    for i in unused
        j = wsample(1:n, tcosts)
        tcosts[j] = 0
        v = x[:,j]
        centers[:,i] = v

        colwise!(ds, SqEuclidean(), v, x)
        tcosts = min(tcosts, ds)
    end
end


type KmeansResult{T<:FloatingPoint}
    centers::Matrix{T}         # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{T}           # costs of the resultant assignments (n)
    counts::Vector{Int}        # number of samples assigned to each cluster (k)
    cweights::Vector{T}        # cluster weights (k)
    total_cost::Float64        # total cost (i.e. objective) (k)
    iterations::Int            # number of elapsed iterations 
    converged::Bool            # whether the procedure converged
end

# core k-means skeleton

function _kmeans!{T<:FloatingPoint}(
    x::Matrix{T},                  # in: sample matrix (d x n)
    w::Union(Nothing, Vector{T}),  # in: sample weights (n)
    centers::Matrix{T},            # in/out: matrix of centers (d x k)
    assignments::Vector{Int},      # out: vector of assignments (n)
    costs::Vector{T},              # out: costs of the resultant assignments (n)
    counts::Vector{Int},           # out: the number of samples assigned to each cluster (k)
    cweights::Vector{T},           # out: the weights of each cluster
    opts::KmeansOpts)              # in: options

    # process options

    tol::Float64 = opts.tol
    max_iters::Int = opts.max_iters
    display::Symbol = opts.display

    displevel = display_level(opts.display)

    # initialize

    k = size(centers, 2)
    to_update = Array(Bool, k) # indicators of whether a center needs to be updated
    unused = Int[]
    num_affected::Int = k # number of centers, to which the distances need to be recomputed

    dmat = pairwise(SqEuclidean(), centers, x)
    dmat = convert(Array{T}, dmat) #Can be removed if one day Distance.result_type(SqEuclidean(), T, T) == T
    update_assignments!(dmat, true, assignments, costs, counts, to_update, unused)
    objv = w == nothing ? sum(costs) : dot(w, costs)

    # main loop
    if displevel >= 2
        @printf "%7s %18s %18s | %8s \n" "Iters" "objv" "objv-change" "affected"
        println("-------------------------------------------------------------")
    end

    t = 0

    converged = false

    while !converged && t < opts.max_iters
        t = t + 1

        # update (affected) centers

        update_centers!(x, w, assignments, to_update, centers, cweights)

        if !isempty(unused)
            repick_unused_centers(x, costs, centers, unused)
        end

        # update pairwise distance matrix

        if !isempty(unused)
            to_update[unused] = true
        end

        if t == 1 || num_affected > 0.75 * k
            pairwise!(dmat, SqEuclidean(), centers, x)
        else
            # if only a small subset is affected, only compute for that subset
            affected_inds = find(to_update)
            dmat_p = pairwise(SqEuclidean(), centers[:, affected_inds], x)
            dmat[affected_inds, :] = dmat_p
        end

        # update assignments

        update_assignments!(dmat, false, assignments, costs, counts, to_update, unused)
        num_affected = sum(to_update) + length(unused)

        # compute change of objective and determine convergence

        prev_objv = objv
        objv = w == nothing ? sum(costs) : dot(w, costs)
        objv_change = objv - prev_objv

        if objv_change > tol
            warn("The objective value changes towards an opposite direction")
        end

        if abs(objv_change) < tol
            converged = true
        end

        # display iteration information (if asked)

        if displevel >= 2
            @printf "%7d %18.6e %18.6e | %8d\n" t objv objv_change num_affected
        end
    end

    if displevel >= 1
        if converged
            println("K-means converged with $t iterations (objv = $objv)")
        else
            println("K-means terminated without convergence after $t iterations (objv = $objv)")
        end
    end

    return KmeansResult(centers, assignments, costs, counts, cweights, float64(objv), t, converged)
end


###########################################################
#
#   Interface functions
#
###########################################################

function check_k(n, k)
    if !(k >=2 && k < n)
        throw( ArgumentError("k must be in [2, n)") )
    end
end

function kmeans!{T<:FloatingPoint}(
    x::Matrix{T},
    centers::Matrix{T},
    opts::KmeansOpts)

    m::Int, n::Int = size(x)
    m2::Int, k::Int = size(centers)
    if m != m2
        throw(ArgumentError("Mismatched dimensions in x and init_centers."))
    end
    check_k(n, k)

    w = opts.weights
    if w != nothing
        if length(w) != size(x, 2)
            throw(ArgumentError("The lenght of w must match the number of columns in x."))
        end
    end

    assignments = zeros(Int, n)
    costs = zeros(T, n)
    counts = Array(Int, k)
    weights = opts.weights
    cweights = Array(T, k)

    if isa(weights, Vector)
        if !(eltype(weights) == T)
            throw(ArgumentError("The element type of weights must be the same as that of samples."))
        end
    end

    _kmeans!(x, weights, centers, assignments, costs, counts, cweights, opts)
end


function kmeans{T<:FloatingPoint}(
    x::Matrix{T},
    init_centers::Matrix{T},
    opts::KmeansOpts)

    kmeans!(x, copy(init_centers), opts)
end

kmeans(x::Matrix, init_centers::Matrix; opts...) = kmeans(x, init_centers, kmeans_opts(;opts...))


function kmeans(x::Matrix, k::Int, opts::KmeansOpts)
    m, n = size(x)
    check_k(n, k)
    init_centers = copyseeds(x, kmpp(x, k))
    kmeans!(x, init_centers, opts)
end

kmeans(x::Matrix, k::Int; opts...) = kmeans(x, k, kmeans_opts(;opts...))

