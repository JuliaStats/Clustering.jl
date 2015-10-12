# K-medoids algorithm


#### Result type

type KmedoidsResult{T} <: ClusteringResult
    medoids::Vector{Int}        # indices of methods (k)
    assignments::Vector{Int}    # assignments (n)
    acosts::Vector{T}           # costs of the resultant assignments (n)
    counts::Vector{Int}         # number of samples assigned to each cluster (k)
    totalcost::Float64          # total assignment cost (i.e. objective) (k)
    iterations::Int             # number of elapsed iterations 
    converged::Bool             # whether the procedure converged
end


#### interface functions

const _kmed_default_init = :kmpp
const _kmed_default_maxiter = 200
const _kmed_default_tol = 1.0e-8
const _kmed_default_display = :none

function kmedoids{T<:Real}(costs::DenseMatrix{T}, k::Integer; 
                           init=_kmed_default_init, 
                           maxiter::Integer=_kmed_default_maxiter, 
                           tol::Real=_kmed_default_tol,
                           display::Symbol=_kmed_default_display)
    # check arguments
    n = size(costs, 1)
    size(costs, 2) == n || error("costs must be a square matrix.")
    k <= n || error("Number of medoids should be less than n.")

    # initialize medoids
    medoids = initseeds_by_costs(init, costs, k)::Vector{Int}
    @assert length(medoids) == k

    # invoke core algorithm
    _kmedoids!(medoids, costs, 
               round(Int, maxiter), tol, display_level(display))
end

function kmedoids!{T<:Real}(costs::DenseMatrix{T}, medoids::Vector{Int};
                            maxiter::Integer=_kmed_default_maxiter, 
                            tol::Real=_kmed_default_tol,
                            display::Symbol=_kmed_default_display)

    # check arguments
    n = size(costs, 1)
    size(costs, 2) == n || error("costs must be a square matrix.")
    length(medoids) <= n || error("Number of medoids should be less than n.")

    # invoke core algorithm
    _kmedoids!(medoids, costs, 
               round(Int, maxiter), tol, display_level(display))
end


#### core algorithm

function _kmedoids!{T<:Real}(medoids::Vector{Int},      # initialized medoids
                             costs::DenseMatrix{T},     # cost matrix
                             maxiter::Int,              # maximum number of iterations
                             tol::Real,                 # tolerable change of objective
                             displevel::Int)            # level of display

    # cost[i, j] is the cost of assigning sample j to the medoid i
    
    n = size(costs, 1)
    k = length(medoids)

    # prepare storage
    acosts = Array(T, n)
    counts = zeros(T, k)
    assignments = Array(Int, n)

    groups = Array(Vector{Int}, k)
    for i = 1:k
        @inbounds groups[i] = Int[]
    end

    # initialize assignments
    tcost, _ = _kmed_update_assignments!(costs, medoids, assignments, groups, acosts, true)

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
            medoids[i] = _find_medoid(costs, groups[i])
        end

        # update assignments
        tcost_pre = tcost
        tcost, ch = _kmed_update_assignments!(costs, medoids, assignments, groups, acosts, false)

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
        acosts,
        counts, 
        tcost,
        t, converged)
end


# update assignments and related quantities
function _kmed_update_assignments!{T}(costs::DenseMatrix{T},        # in: (n, n)
                                      medoids::AbstractVector{Int}, # in: (k,)
                                      assignments::Vector{Int},     # out: (n,)
                                      groups::Vector{Vector{Int}},  # out: (k,)
                                      acosts::Vector{T},            # out: (n,)
                                      isinit::Bool)                 # in    
    n = size(costs, 1)
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
        mv = costs[medoids[1], j]

        for i = 2:k
            v = costs[medoids[i], j]
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

        acosts[j] = mv
        tcost += mv
        push!(groups[p], j)
    end

    return (tcost, ch)::Tuple{Float64, Int}
end


# find medoid for a given group
#
# TODO: faster way without creating temporary arrays
function _find_medoid(costs::DenseMatrix, grp::Vector{Int})
    @assert !isempty(grp)
    p = indmin(sum(costs[grp, grp], 2))
    return grp[p]::Int
end


