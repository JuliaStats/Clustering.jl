# Affinity propagation
#
#   Reference:
#       Clustering by Passing Messages Between Data Points.
#       Brendan J. Frey and Delbert Dueck
#       Science, vol 315, pages 972-976, 2007.
#

#### Interface

"""
The result of affinity propagation clustering ([`affinityprop`](@ref)).

# Fields
 * `exemplars::Vector{Int}`: indices of *exemplars* (cluster centers)
 * `assignments::Vector{Int}`: cluster assignments for each data point
 * `iterations::Int`: number of iterations executed
 * `converged::Bool`: converged or not
"""
mutable struct AffinityPropResult <: ClusteringResult
    exemplars::Vector{Int}      # indexes of exemplars (centers)
    assignments::Vector{Int}    # assignments for each point
    counts::Vector{Int}         # number of data points in each cluster
    iterations::Int             # number of iterations executed
    converged::Bool             # converged or not
end

const _afp_default_maxiter = 200
const _afp_default_damp = 0.5
const _afp_default_tol = 1.0e-6
const _afp_default_display = :none

"""
    affinityprop(S::DenseMatrix; [maxiter=200], [tol=1e-6], [damp=0.5],
                 [display=:none])

Perform affinity propagation clustering based on a similarity matrix `S`.

``S_{ij}`` (``i ≠ j``) is the similarity (or the negated distance) between
the ``i``-th and ``j``-th points, ``S_{ii}`` defines the *availability*
of the ``i``-th point as an *exemplar*.

Returns an instance of [`AffinityPropResult`](@ref).

# Method parameters
 - `damp::Real`: the dampening coefficient, ``0 ≤ \\mathrm{damp} < 1``.
   Larger values indicate slower (and probably more stable) update.
   ``\\mathrm{damp} = 0`` disables dampening.
 - `maxiter`, `tol`, `display`: see [common options](@ref common_options)

# Notes
The implementations is based on the following paper:

> Brendan J. Frey and Delbert Dueck. *Clustering by Passing Messages
> Between Data Points.* Science, vol 315, pages 972-976, 2007.
"""
function affinityprop(S::DenseMatrix{T};
                      maxiter::Integer=_afp_default_maxiter,
                      tol::Real=_afp_default_tol,
                      damp::Real=_afp_default_damp,
                      display::Symbol=_afp_default_display) where T<:AbstractFloat

    # check arguments
    n = size(S, 1)
    size(S, 2) == n || error("S must be a square matrix.")
    n >= 2 || error("the number of data points must be at least 2.")
    tol > 0 || error("tol must be a positive value.")
    0 <= damp < 1 || error("damp must be a non-negative real value below 1.")

    # invoke core implementation
    _affinityprop(S, round(Int, maxiter), tol, convert(T, damp), display_level(display))
end


#### Implementation

function _affinityprop(S::DenseMatrix{T},
                       maxiter::Int,
                       tol::Real,
                       damp::T,
                       displevel::Int) where T<:AbstractFloat
    n = size(S, 1)
    n2 = n * n

    # initialize messages
    R = zeros(T, n, n)  # responsibilities
    A = zeros(T, n, n)  # availabilities

    # prepare storages
    Rt = Matrix{T}(undef, n, n)
    At = Matrix{T}(undef, n, n)

    if displevel >= 2
        @printf "%7s %12s | %8s \n" "Iters" "objv-change" "exemplars"
        println("-----------------------------------------------------")
    end

    t = 0
    converged = false
    while !converged && t < maxiter
        t += 1

        # compute new messages
        _afp_compute_r!(Rt, S, A)
        _afp_dampen_update!(R, Rt, damp)

        _afp_compute_a!(At, R)
        _afp_dampen_update!(A, At, damp)

        # determine convergence
        ch = max(Linfdist(A, At), Linfdist(R, Rt)) / (one(T) - damp)
        converged = (ch < tol)

        if displevel >= 2
            # count the number of exemplars
            ne = _afp_count_exemplars(A, R)
            @printf("%7d %12.4e | %8d\n", t, ch, ne)
        end
    end

    # extract exemplars and assignments
    exemplars = _afp_extract_exemplars(A, R)
    assignments, counts = _afp_get_assignments(S, exemplars)

    if displevel >= 1
        if converged
            println("Affinity propagation converged with $t iterations: $(length(exemplars)) exemplars.")
        else
            println("Affinity propagation terminated without convergence after $t iterations: $(length(exemplars)) exemplars.")
        end
    end

    # produce output struct
    return AffinityPropResult(exemplars, assignments, counts, t, converged)
end


# compute responsibilities
function _afp_compute_r!(R::Matrix{T}, S::DenseMatrix{T}, A::Matrix{T}) where T
    n = size(S, 1)

    I1 = Vector{Int}(undef, n)  # I1[i] is the column index of the maximum element in (A+S)[i,:]
    Y1 = Vector{T}(undef, n)    # Y1[i] is the maximum element in (A+S)[i,:]
    Y2 = Vector{T}(undef, n)    # Y2[i] is the second maximum element in (A+S)[i,:]

    # Find the first and second maximum elements along each row
    @inbounds for i = 1:n
        v1 = A[i,1] + S[i,1]
        v2 = A[i,2] + S[i,2]
        if v1 > v2
            I1[i] = 1
            Y1[i] = v1
            Y2[i] = v2
        else
            I1[i] = 2
            Y1[i] = v2
            Y2[i] = v1
        end
    end
    @inbounds for j = 3:n, i = 1:n
        v = A[i,j] + S[i,j]
        if v > Y2[i]
            if v > Y1[i]
                Y2[i] = Y1[i]
                I1[i] = j
                Y1[i] = v
            else
                Y2[i] = v
            end
        end
    end

    # compute R values
    @inbounds for j = 1:n, i = 1:n
        mv = (j == I1[i] ? Y2[i] : Y1[i])
        R[i,j] = S[i,j] - mv
    end

    return R
end

# compute availabilities
function _afp_compute_a!(A::Matrix{T}, R::Matrix{T}) where T
    n = size(R, 1)
    z = zero(T)
    for j = 1:n
        @inbounds rjj = R[j,j]

        # compute s <- sum_{i \ne j} max(0, R(i,j))
        s = z
        for i = 1:n
            if i != j
                @inbounds r = R[i,j]
                if r > z
                    s += r
                end
            end
        end

        for i = 1:n
            if i == j
                @inbounds A[i,j] = s
            else
                @inbounds r = R[i,j]
                u = rjj + s
                if r > z
                    u -= r
                end
                A[i,j] = ifelse(u < z, u, z)
            end
        end
    end
    return A
end

# dampen update
function _afp_dampen_update!(x::Array{T}, xt::Array{T}, damp::T) where T
    ct = one(T) - damp
    for i = 1:length(x)
        @inbounds x[i] = ct * xt[i] + damp * x[i]
    end
    return x
end

# count the number of exemplars
function _afp_count_exemplars(A::Matrix, R::Matrix)
    n = size(A,1)
    c = 0
    for i = 1:n
        @inbounds if A[i,i] + R[i,i] > 0
            c += 1
        end
    end
    return c
end

# extract all exemplars
function _afp_extract_exemplars(A::Matrix, R::Matrix)
    n = size(A,1)
    r = Int[]
    for i = 1:n
        @inbounds if A[i,i] + R[i,i] > 0
            push!(r, i)
        end
    end
    return r
end

# get assignments
function _afp_get_assignments(S::DenseMatrix, exemplars::Vector{Int})
    n = size(S, 1)
    k = length(exemplars)
    Se = S[:, exemplars]
    a = Vector{Int}(undef, n)
    cnts = zeros(Int, k)
    for i = 1:n
        p = 1
        v = Se[i,1]
        for j = 2:k
            s = Se[i,j]
            if s > v
                v = s
                p = j
            end
        end
        a[i] = p
    end
    for i = 1:k
        a[exemplars[i]] = i
    end
    for i = 1:n
        @inbounds cnts[a[i]] += 1
    end
    return (a, cnts)
end
