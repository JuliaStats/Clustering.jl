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
    counts      :: Vector{Int} # number of samples assigned to each cluster
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
            x    = X[:,i]
            d_ic = map(i->dot(x-μ[:,i], x-μ[:,i]), [1:k]) # distance from cluster c
            j    = indmin(d_ic) # index of best cluster
            if d_ic[j] > λ
                # create a new cluster
                k   += 1
                assignments[i] = k
                μ    = [μ x]
                n_updated += 1
            else
                # assign to best
                n_updated += assignments[i] != j
                assignments[i] = j
            end
            score += d_ic[j] # add score
        end

        if displevel >= 2
            # display iteration information (if asked)
            @printf("%7d %7d %18.6e %18.6e | %8d\n", n_iter, k, score, abs(score - score_prev), n_updated)
        end

        # check for convergence
        if abs(score - score_prev) < tol
            converged = true
        else
            # generate clusters
            L = [Int[] for i=1:k]
            for (i,z_) in enumerate(assignments)
                push!(L[z_], i)
            end

            # remove empty clusters
            filter!(l->!isempty(l), L)
            k = length(L)

            # update means
            μ = Array(T, m, k)
            for (i,l) in enumerate(L)
                μ[:,i] = sum(X[:,l], 2) ./ length(l)
            end
        end
    end

    # -------------------

    map!(i->norm(X[:,i]-μ[:,assignments[i]],2), costs, [1:n])
    counts    = map(l->length(l), L)
    totalcost = sum(costs)

    if displevel >= 1
        if converged
            println("DP-means converged in $n_iter iterations producing $k clusters (objv = $totalcost)")
        else
            println("DP-means terminated without convergence after $n_iter iterations producing $k clusters  (objv = $totalcost)")
        end
    end

    return DPmeansResult(k, λ, μ, assignments, costs, counts, totalcost, n_iter, converged)
end