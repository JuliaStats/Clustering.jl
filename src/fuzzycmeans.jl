# Fuzzy C means algorithm

## Interface

"""
    FuzzyCMeansResult{T<:AbstractFloat}

The output of [`fuzzy_cmeans`](@ref) function.

# Fields
- `centers::Matrix{T}`: the ``d×C`` matrix with columns being the
  centers of resulting fuzzy clusters
- `weights::Matrix{Float64}`: the ``n×C`` matrix of assignment weights
  (``\\mathrm{weights}_{ij}`` is the weight (probability) of assigning
  ``i``-th point to the ``j``-th cluster)
- `iterations::Int`: the number of executed algorithm iterations
- `converged::Bool`: whether the procedure converged
"""
struct FuzzyCMeansResult{T<:AbstractFloat}
    centers::Matrix{T}          # cluster centers (d x C)
    weights::Matrix{Float64}    # assigned weights (n x C)
    iterations::Int             # number of elapsed iterations
    converged::Bool             # whether the procedure converged
end

nclusters(R::FuzzyCMeansResult) = size(R.centers, 2)

wcounts(R::FuzzyCMeansResult) = dropdims(sum(R.weights, dims=2), dims=2)

## Utility functions

function update_weights!(weights, data, centers, fuzziness, dist_metric)
    pow = 2.0/(fuzziness-1)
    nrows, ncols = size(weights)
    dists = pairwise(dist_metric, data, centers, dims=2)
    for i in 1:nrows
        for j in 1:ncols
            den = 0.0
            for k in 1:ncols
                den += (dists[i,j]/dists[i,k])^pow
            end
            weights[i,j] = 1.0/den
        end
    end
end

function update_centers!(centers, data, weights, fuzziness)
    nrows, ncols = size(weights)
    T = eltype(centers)
    for j in 1:ncols
        num = zeros(T, size(data,1))
        den = zero(T)
        for i in 1:nrows
            δm = weights[i,j]^fuzziness
            num += δm * data[:,i]
            den += δm
        end
        centers[:,j] = num/den
    end
end

const _fcmeans_default_maxiter = 100
const _fcmeans_default_tol = 1.0e-3
const _fcmeans_default_display = :none

"""
    fuzzy_cmeans(data::AbstractMatrix, C::Integer, fuzziness::Real;
                 [dist_metric::SemiMetric], [...]) -> FuzzyCMeansResult

Perform Fuzzy C-means clustering over the given `data`.

# Arguments
 - `data::AbstractMatrix`: ``d×n`` data matrix. Each column represents
   one ``d``-dimensional data point.
 - `C::Integer`: the number of fuzzy clusters, ``2 ≤ C < n``.
 - `fuzziness::Real`: clusters fuzziness (``μ`` in the
   [mathematical formulation](@ref fuzzy_cmeans_def)), ``μ > 1``.

Optional keyword arguments:
 - `dist_metric::SemiMetric` (defaults to `Euclidean`): the `SemiMetric` object
    that defines the distance between the data points
 - `maxiter`, `tol`, `display`, `rng`: see [common options](@ref common_options)
"""
function fuzzy_cmeans(
    data::AbstractMatrix{<:Real},
    C::Integer,
    fuzziness::Real;
    maxiter::Integer = _fcmeans_default_maxiter,
    tol::Real = _fcmeans_default_tol,
    dist_metric::SemiMetric = Euclidean(),
    display::Symbol = _fcmeans_default_display,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    nrows, ncols = size(data)
    2 <= C < ncols || throw(ArgumentError("C must have 2 <= C < n=$ncols ($C given)"))
    1 < fuzziness || throw(ArgumentError("fuzziness must be greater than 1 ($fuzziness given)"))

    _fuzzy_cmeans(data, C, fuzziness,
                  maxiter, tol, dist_metric, display_level(display), rng)
end

## Core implementation

function _fuzzy_cmeans(
    data::AbstractMatrix{T},                        # data matrix
    C::Integer,                                     # total number of classes
    fuzziness::Real,                                # fuzziness
    maxiter::Int,                                   # maximum number of iterations
    tol::Real,                                      # tolerance
    dist_metric::SemiMetric,                        # metric to calculate distance
    displevel::Int,                                 # the level of display
    rng::AbstractRNG                                # RNG object
) where T<:Real

    nrows, ncols = size(data)

    # Initialize weights randomly
    weights = rand(rng, Float64, ncols, C)
    weights ./= sum(weights, dims=2)

    centers = zeros(T, (nrows, C))
    prev_centers = identity.(centers)

    δ = Inf
    iter = 0

    if displevel >= 2
        @printf "%7s %18s\n" "Iters" "center-change"
        println("----------------------------")
    end

    while iter < maxiter && (iter <= 1 || δ > tol) # skip tol test for iter=1 since prev_centers are not relevant
        update_centers!(centers, data, weights, fuzziness)
        update_weights!(weights, data, centers, fuzziness, dist_metric)
        δ = maximum(colwise(dist_metric, prev_centers, centers))
        copyto!(prev_centers, centers)
        iter += 1
        if displevel >= 2
            @printf("%7d %18.6e\n", iter, δ)
        end
    end

    if δ <= tol
        if displevel >= 1
            @info "Fuzzy C-means converged with $iter iterations (δ = $δ)"
        end
    else
        @warn "Fuzzy C-means terminated without convergence after $iter iterations (δ = $δ)"
    end

    FuzzyCMeansResult(centers, weights, iter, δ <= tol)
end

function Base.show(io::IO, result::FuzzyCMeansResult)
    d, C = size(result.centers)
    n, iter = size(result.weights, 1), result.iterations
    print(io, "FuzzyCMeansResult: $C clusters for $n points in $d dimensions ")
    if result.converged
        print(io, "(converged in $iter iterations)")
    else
        print(io, "(failed to converge in $iter iterations)")
    end
end
