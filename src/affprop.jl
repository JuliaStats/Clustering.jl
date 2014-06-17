############################################################
# Affinity Propagation
############################################################

type AffinityPropagationOpts
    max_iter     ::Int
    n_stop_check ::Int
    damp         ::Float64
    display      ::Symbol
end

function AffinityPropagationOpts(;
    max_iter::Integer = 500,    # max number of iterations
    n_stop_check::Integer = 10, # stop if exemplars not changed for this number of iterations
    damp::FloatingPoint = 0.5,  # damping factor for message updating, 0 means no damping
    display::Symbol = :iter     # verboseness: :iter, :final, :none
    )

    AffinityPropagationOpts(int(max_iter), int(n_stop_check), float(damp), display)
end

type AffinityPropagationResult
    exemplar_index ::Vector{Int} # index for exemplars (centers)
    assignments    ::Vector{Int} # assignments for each point
    iterations     ::Int         # number of iterations executed
    converged      ::Bool        # converged or not
end

function affinity_propagation{T<:FloatingPoint}(S::Matrix{T}, opts::AffinityPropagationOpts)
    n = size(S, 1)
    converged = false

    exemplars = falses(n, opts.n_stop_check)
    psi = zeros(eltype(S), n, n)
    phi = zeros(eltype(S), n, n)

    n_iter = 0
    IC = nothing

    for n_iter = 1:opts.max_iter
        affprop_update_message!(psi, phi, S, n, opts)

        IC = diag(psi) + diag(phi) .> 0
        exemplars[:, mod(n_iter-1, opts.n_stop_check)+1] = IC

        if opts.display == :iter
            @printf("%7d: %3d exemplars identified\n", n_iter, countnz(IC))
        end

        if n_iter > opts.n_stop_check
            converged = true
            for i = 1:opts.n_stop_check
                if IC != exemplars[:, i]
                    converged = false
                    break
                end
            end
            if converged
                break
            end
        end
    end

    assignment, exemplar_index = affprop_decide_assignment(phi, S, n, IC)
    if opts.display == :iter || opts.display == :final
        if converged
            println("affinity propagation converged with $n_iter iterations")
        else
            println("affinity propagation terminated without convergence after $n_iter iterations")
        end
    end

    return AffinityPropagationResult(exemplar_index, assignment, n_iter, converged)
end

affinity_propagation(x::Matrix; opts...) = affinity_propagation(x, AffinityPropagationOpts(;opts...))

############################################################
# Message Updating
#
# Loopy message passing sometimes can be tricky. Parallel
# updating vs. in-place updating; damping vs. no-damping;
# as well as the order of message updating can affect the
# convergence behavior drastically.
############################################################
function affprop_update_message!{T<:FloatingPoint}(psi::Matrix{T}, phi::Matrix{T}, S::Matrix{T}, n, opts::AffinityPropagationOpts)
    # update phi
    for i = 1:n
        SM = S[i,:] + psi[i,:]
        I = indmax(SM)
        Y = SM[I]
        SM[I] = -Inf
        Y2 = maximum(SM)
        val = S[i,:] .- Y;
        val[I] = S[i,I] - Y2;
        phi[i,:] = opts.damp*phi[i,:] + (1-opts.damp)*val
    end

    # update psi
    for j = 1:n
        RP = phi[:,j]
        idx = 1:n .!= j
        RP[idx] = max(RP[idx], 0)
        val = sum(RP) .- RP;
        val[idx] = min(val[idx], 0)
        psi[:,j] = opts.damp*psi[:,j] + (1 - opts.damp)*val
    end
end

function affprop_decide_assignment{T<:FloatingPoint}(phi::Matrix{T}, S::Matrix{T}, n::Integer, exemplar::BitArray{1})
    assignment = zeros(Int, n)
    Iexp = find(exemplar)

    for i = 1:n
        I = indmax(phi[i,Iexp] + S[i,Iexp])
        assignment[i] = Iexp[I]
    end

    # make sure exemplars are assigned to themselves
    assignment[Iexp] = Iexp

    return assignment, Iexp
end
