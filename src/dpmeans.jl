# DP-means algorithm
# source: "Revisiting k-means: New Algorithms via Bayesian Nonparametrics"
#          by Brian Kulis & Michael Jordan

#### Interface

type DPmeansResult{T<:FloatingPoint} <: ClusteringResult
    k           :: Int         # number of clusters
    λ           :: T           # cluster penalty parameter
    centers     :: Matrix{T}   # cluster centers (d x k)
    assignments :: Vector{Int} # assignments (n)
    costs       :: Vector{T}   # costs of the resultant assignments (n)
    counts      :: Vector{Int} # number of samples assigned to each cluster (k)
    totalcost   :: T           # total cost of the clustering
    iterations  :: Int         # number of elapsed iterations
    converged   :: Bool        # whether the procedure converged
end

const _dpmeans_default_init    = :dpmpp
const _dpmeans_default_maxiter = 100
const _dpmeans_default_tol     = 1.0e-6
const _dpmeans_default_display = :none

function dpmeans{T<:FloatingPoint}(
    X       :: Matrix{T}, # input data, [m×n]; columns are x-vectors
    λ       :: Real;      # clustering penalty parameter
    maxiter :: Integer = _dpmeans_default_maxiter,
    tol     :: Real    = _dpmeans_default_tol,
    display :: Symbol = _dpmeans_default_display
    )
    
    m,n = size(X)
    assignments = zeros(Int, n)
    costs       = zeros(T,   n)

    _dpmeans!(X, convert(T,λ), assignments, costs, int(maxiter), tol, display_level(display))
end


#### Core implementation

# core dp-means algorithm
function _dpmeans!{T<:FloatingPoint}(
    X           :: Matrix{T},  # in: input data, (d x n); columns are x-vectors
    λ           :: T,          # in: cluster penalty parameter
    assignments :: Vector{Int},# out: vector of assignments (n)
    costs       :: Vector{T},  # out: vector of assignment costs (n)
    maxiter     :: Integer,    # in: maximum number of iterations
    tol         :: Real,       # in: tolerance of change at convergence (tolerance is on squared norm, not on true norm)
    displevel   :: Int         # in: the level of display
    )

    # out: 
    #    L - cluster list ::Vector{Vector{Int}} (k)

    # -------------------
    # init 
    const m,n  = size(X)
    k          = 1                     # cluster count
    L          = Array(Vector{Int}, 1) # cluster list
    L[1]       = [1:n]                 # assign all variables to one cluster
    μ          = Array(T, m, 1)        # mean list
    μ[:,1]     = mean(X,2)             # initialize to global mean
    score      = Inf                   # current algorithm score
    n_iter     = 0                     # iteration counter
    fill!(assignments, one(Int))       # init cluster indicators
    
    # -------------------
    # repeat until convergence

    if displevel >= 2
            @printf "%7s %7s %18s %18s | %8s \n" "Iters" "Clstrs" "objv" "objv-change" "affected"
            println("----------------------------------------------------------------")
    end

    converged = false
    while !converged && n_iter < maxiter

        n_iter += 1
        score_prev = score

        # println("Iteration: ", n_iter)

        # update centers
        score = 0.0
        n_updated = 0
        for i = 1 : n
            x = X[:,i]

            # identify best cluster and best cluster distance
            best_cluster_dist  = Inf
            best_cluster_index = -1
            for j = 1 : k
                dist = 0.0 # dot product
                for l = 1 : m
                    dist += (x[l] - μ[l,j])*(x[l] - μ[l,j])
                end
                if dist < best_cluster_dist
                    best_cluster_dist, best_cluster_index = dist, j
                end
            end

            if best_cluster_dist > λ
                # create a new cluster
                k   += 1
                assignments[i] = k
                μ    = [μ x]
                n_updated += 1
            else
                # assign to best
                n_updated += (assignments[i] != best_cluster_index)
                assignments[i] = best_cluster_index
            end
            score += best_cluster_dist # add score
        end

        if displevel >= 2
            # display iteration information (if asked)
            @printf("%7d %7d %18.6e %18.6e | %8d\n", n_iter, k, score, abs(score - score_prev), n_updated)
        end

        # check for convergence
        if abs(score - score_prev) < tol
            converged = true
        else

            # recompute means
            μ = zeros(T, m, k)
            assignment_mat = falses(n,k)
            counts = zeros(Int, k)
            for (i,a) in enumerate(assignments)
                assignment_mat[i,a] = true
                counts[a] += 1
            end

            # identify empty clusters
            good_clusters = falses(k)
            for i = 1 : k
                if counts[i] != 0
                    good_clusters[i] = true
                    μ[:,i] = sum(X[:,assignment_mat[:,i]], 2) ./ counts[i]
                end
            end

            μ = μ[:,good_clusters]
            k = size(μ,2)
        end
    end

    # -------------------

    # compute costs and counts
    totalcost = zero(T)
    counts = zeros(Int, k)
    for i = 1 : n
        cost = zero(T)
        assignment = assignments[i]
        for j = 1 : m
            cost += (X[j,i] - μ[j,assignment])*(X[j,i] - μ[j,assignment])
        end
        cost = sqrt(cost)
        costs[i] = cost
        totalcost += cost
        counts[assignment] += 1
    end

    if displevel >= 1
        if converged
            println("DP-means converged in $n_iter iterations producing $k clusters (objv = $totalcost)")
        else
            println("DP-means terminated without convergence after $n_iter iterations producing $k clusters  (objv = $totalcost)")
        end
    end

    return DPmeansResult(k, λ, μ, assignments, costs, counts, totalcost, n_iter, converged)
end