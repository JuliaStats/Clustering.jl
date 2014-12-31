# PCCA+ algorithm

#### Interface

type PccapResult <: ClusteringResult
    assignments::Vector # discrete cluster assignment
    counts::Vector
    chi::Matrix         # fuzzy cluster assignment
end

function pccap(P, n; pi=nothing)
    if pi == nothing
        pi = abs(vec((Array{Float64})(eigs(P';nev=1)[2])))
        pi = pi / sum(pi)                     # => first col of X is one
    end
    
    X = schurvectors(P, pi, n)
    A = getA(X)
    chi = X*A

    assignments = vec(mapslices(indmax,chi,2))
    counts = hist(assignments)[2]
    return PccapResult(assignments, counts, chi)
end

function schurvectors(P, pi, n)
    Pw = diagm(sqrt(pi))*P*diagm(1./sqrt(pi)) # rescale to keep markov property
    Sw = schurfact!(Pw)                       # returns orthonormal vecs by def
    Xw = selclusters!(Sw, n)
    X  = diagm(1./sqrt(pi)) * Xw              # scale back
    if X[1,1]<0
        X = -X
    end
    X
end

# select the schurvectors corresponding to the n abs-largest eigenvalues
function selclusters!(S, n)
    ind = sortperm(abs(S[:values]), rev=true) # get indices for largest eigenvalues
    select = zeros(Int, size(ind))            # create selection vector
    select[ind[1:n]] = 1
    S = ordschur!(S, select)                  # reorder selected vectors to the left
    X = S[:vectors][:,1:n]                    # select first n vectors
end

function getA(X)
    A0=feasible(guess(X),X)
    A=opt(A0, X)
end

# compute initial guess based on indexmap
guess(X) = inv(X[indexmap(X), :])

function indexmap(X)
    # get indices of rows of X to span the largest simplex
    rnorm(x) = sqrt(sumabs2(x,2))
    ind=zeros(size(X,2))
    for j in 1:length(ind)
        rownorm=rnorm(X)
        # store largest row index
        ind[j]=indmax(rownorm)
        if j == 1
            # translate to origin
            X=broadcast(-, X, X[ind[1],:])
        else
            # remove subspace
            X=X/rownorm[ind[j]]
            vt=X[ind[j],:]
            X=X-X*vt'*vt
        end
    end
    return ind
end

function feasible(A,X)
    A[:,1] = -sum(A[:,2:end], 2)
    A[1,:] = -minimum(X[:,2:end] * A[2:end,:], 1)
    A / sum(A[1,:])
end

function opt(A0,X)

    I1(A) = sum(maximum(X*A,1))

    function transform(A)
      # cut out the fixed part
      cA = A[2:end, 2:end]
      # flatten matrix to vector for use in optimize
      return reshape(cA,prod(size(cA)))
    end

    function transformback(tA)
      # reshape back to matrix
      cA = reshape(tA, size(A0,1)-1, size(A0,2)-1)
      # unite with the fixed part
      A = A0
      A[2:end,2:end] = cA
      return A
    end

    obj(tA) = -I1(feasible(transformback(tA), X))
    result = optimize(obj, transform(A0), method = :nelder_mead)
    return feasible(transformback(result.minimum), X)
end
