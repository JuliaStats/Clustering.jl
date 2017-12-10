## hclust.jl (c) 2014, 2017 David A. van Leeuwen
## Hierarchical clustering, similar to R's hclust()

## Algorithms are based upon C. F. Olson, Parallel Computing 21 (1995) 1313--1325.

## This is also in types.jl, but that is not read...
"""
Hierarchical clustering of the data returned by `hclust()`.
The data hierarchy is defined by the `merges` matrix:
 - each row specifies which subtrees (referenced by their IDs) are merged into a higher-level subtree
 - negative subtree `id` denotes leaf node and corresponds to the datapoint
   at position `-id`
 - positive `id` denotes nontrivial subtree:
   the row `merges[id, :]` specifies its left and right subtrees,
   and `heights[id]` -- its height.

This type mostly follows R's `hclust` class.
"""
struct Hclust{T<:Real}
    merges::Matrix{Int} # the tree merge sequence. 1st column: left subtree, 2nd column: right subtree
    heights::Vector{T}  # subtrees heights (aggregated distance between its elements)
    order::Vector{Int}  # the order of datapoint (leaf node) indices in the final tree
    linkage::Symbol     # subtree distance type (cluster linkage)
end

nmerges(h::Hclust) = length(h.heights)  # number of tree merges
nnodes(h::Hclust) = length(h.order)     # number of datapoints (leaf nodes)
height(h::Hclust) = isempty(h.heights) ? typemin(eltype(h.heights)) : last(h.heights)

function assertdistancematrix(d::AbstractMatrix)
    nr, nc = size(d)
    nr == nc || throw(DimensionMismatch("Distance matrix should be square."))
    issymmetric(d) || throw(ArgumentError("Distance matrix should be symmetric."))
end

## R's order of trees
_isrordered(i::Integer, j::Integer) =
    i < 0 && j < 0 && i > j ||  # leaves (datapoints) are sorted in ascending order
    i > 0 && j > 0 && i < j ||  # if i-th tree was created before j-th one, it goes first
    i < 0 && j > 0              # leaves go before trees

## This seems to work like R's implementation, but it is extremely inefficient
## This probably scales O(n^3) or worse. We can use it to check correctness
function hclust_n3(d::AbstractMatrix, linkage::Function)
    assertdistancematrix(d)
    T = eltype(method(d, 1:0, 1:0))
    mr = Int[]                  # min row
    mc = Int[]                  # min col
    h = T[]                     # height
    n = size(d,1)               # number of datapoints (leaf nodes)
    node2cl = collect(-(1:n))   # datapoint to tree attribution, initially all leaves
    next = 1                    # next cluster label
    while next < n
        NNmindist = typemax(T)
        NNi = NNj = 0           # indices of nearest neighbors clusters
        cl = unique(node2cl)    # active tree ids
        mask = falses(n)
        for j in eachindex(cl)  # loop over for lower triangular indices, i>j
            clj = cl[j]
            cols = node2cl .== clj
            for i in (j+1):length(cl)
                cli = cl[i]
                rows = node2cl .== cli
                dist = linkage(d, rows, cols) # very expensive
                if (NNi == 0) || (dist < NNmindist)
                    NNmindist = dist
                    NNi = cli
                    NNj = clj
                    mask .= cols .| rows
                end
            end
        end
        if !_isrordered(NNi, NNj)
            NNi, NNj = NNj, NNi
        end
        push!(mr, NNi)
        push!(mc, NNj)
        push!(h, NNmindist)
        node2cl[mask] .= next
        next += 1
    end
    hcat(mr, mc), h
end

# nearest neighbor to i-th node given symmetric distance matrix d;
# returns 0 if no nearest neighbor (1×1 matrix)
function nearest_neighbor(d::AbstractMatrix, i::Integer, N::Integer=size(d, 1))
    (N <= 1) && return 0, NaN
    # initialize with the first non-i node
    @inbounds if i > 1
        NNi = 1
        NNdist = d[NNi, i]
    else
        NNi = 2
        NNdist = d[i, NNi]
    end
    @inbounds for j in (NNi+1):(i-1)
        if NNdist > d[j, i]
            NNi = j
            NNdist = d[j, i]
        end
    end
    @inbounds for j in (i+1):N
        if NNdist > d[i, j]
            NNi = j
            NNdist = d[i, j]
        end
    end
    return NNi, NNdist
end

## Efficient single link algorithm, according to Olson, O(n^2), fig 2.
## Verified against R's implementation, correct, and about 2.5 x faster
## For each i < j compute D(i,j) (this is already given)
## For each 0 < i ≤ n compute Nearest Neighbor NN(i)
## Repeat n-1 times
##   find i,j that minimize D(i,j)
##   merge clusters i and j
##   update D(i,j) and NN(i) accordingly
function hclust_minimum(ds::AbstractMatrix{T}) where T<:Real
    d = Matrix(ds)      # active trees distances, only upper (i < j) is used
    n = size(d,1)       # number of points (leaf nodes)
    ## For each 0 < i ≤ n compute Nearest Neighbor NN[i]
    NN = [nearest_neighbor(d, i, n)[1] for i in 1:n]
    ## the main loop
    mleft = Vector{Int}()       # merged left tree
    mright = Vector{Int}()      # merged right tree
    h = Vector{T}()             # tree height
    trees = collect(-(1:n))     # indices of active trees, initialized to all leaves
    while length(trees) > 1     # O(n)
        # find a pair of nearest trees, i and j
        i = 1
        NNmindist = i < NN[i] ? d[i, NN[i]] : d[NN[i], i]
        for k in 2:length(NN)   # O(n)
            @inbounds dist = k < NN[k] ? d[k,NN[k]] : d[NN[k],k]
            if dist < NNmindist
                NNmindist = dist
                i = k
            end
        end
        j = NN[i]
        if i > j
            i, j = j, i     # make sure i < j
        end
        ## update result, compatible to R's order.  It must be possible to do this simpler than this...
        @inbounds mi = trees[i]
        @inbounds mj = trees[j]
        if !_isrordered(mi, mj)
            mi, mj = mj, mi
        end
        push!(mleft, mi)
        push!(mright, mj)
        push!(h, NNmindist)
        trees[i] = length(h) # assign new id to the merged tree
        ## update d, split in ranges k<i, i<k<j, j<k≤nc
        for k in 1:(i-1)         # k < i
            @inbounds if d[k,i] > d[k,j]
                d[k,i] = d[k,j]
            end
        end
        for k in (i+1):(j-1)     # i < k < j
            @inbounds if d[i,k] > d[k,j]
                d[i,k] = d[k,j]
            end
        end
        for k in (j+1):nc        # j < k ≤ nc
            @inbounds if d[i,k] > d[j,k]
                d[i,k] = d[j,k]
            end
        end
        # reassign last tree to position j
        last_tree = length(trees)
        if j < last_tree
            trees[j] = trees[last_tree]
            NN[j] = NN[last_tree]
            ## move the last row/col into j
            for k in 1:(j-1)     # k < j ≤ nc
                @inbounds d[k,j] = d[k,nc]
            end
            for k in (j+1):(nc-1)# j < k < nc
                @inbounds d[j,k] = d[k,nc]
            end
        end
        pop!(NN)
        pop!(trees)
        ## update NN[k]
        for k in eachindex(NN)
            if NN[k] == j        # j is merged into i (only valid for the min!)
                NN[k] = i
            elseif NN[k] == last_tree # last_tree is moved into j
                NN[k] = j
            end
        end
        ## finally we need to update NN[i], because it was nearest to j
        NNmindist = typemax(T)
        NNi = 0
        for k in 1:(i-1)
            @inbounds if (NNi == 0) || (d[k,i] < NNmindist)
                NNmindist = d[k,i]
                NNi = k
            end
        end
        for k in (i+1):length(trees)
            @inbounds if (NNi == 0) || (d[i,k] < NNmindist)
                NNmindist = d[i,k]
                NNi = k
            end
        end
        NN[i] = NNi
#        for n in NN[1:nc] print(n, " ") end; println()
    end
    return hcat(mr, mc), h
end

## functions to compute maximum, minimum, mean for just a slice of an array
## FIXME: method(view(d, cl1, cl2)) would be much more generic, but it leads to extra allocations

function slicemaximum(d::AbstractMatrix, cl1::AbstractVector{Int}, cl2::AbstractVector{Int})
    maxdist = typemin(eltype(d))
    @inbounds for j in cl2, i in cl1
        if d[i,j] > maxdist
            maxdist = d[i,j]
        end
    end
    maxdist
end

function sliceminimum(d::AbstractMatrix, cl1::AbstractVector{Int}, cl2::AbstractVector{Int})
    mindist = typemax(eltype(d))
    @inbounds for j in cl2, i in cl1
        if d[i,j] < mindist
            mindist = d[i,j]
        end
    end
    mindist
end

function slicemean(d::AbstractMatrix, cl1::AbstractVector{Int}, cl2::AbstractVector{Int})
    s = zero(eltype(d))
    @inbounds for j in cl2
        sj = zero(eltype(d))
        for i in cl1
            sj += d[i,j]
        end
        s += sj
    end
    s / (length(cl1)*length(cl2))
end

## reorders the tree merges by the height of resulting trees
## (to be compatible with R's hclust())
function rorder!(mr, mc, h)
    o = sortperm(h)
    io = invperm(o)
    for i in 1:length(mr)
        if mr[i] > 0
            mr[i] = io[mr[i]]
        end
        if mc[i] > 0
            mc[i] = io[mc[i]]
        end
        if !_isrordered(mr[i], mc[i])
            mr[i], mc[i] = mc[i], mr[i]
        end
    end
    return o
end

## Another nearest neighbor algorithm, for reducible metrics
## From C. F. Olson, Parallel Computing 21 (1995) 1313--1325, fig 5
## Verfied against R implementation for mean and maximum, correct but ~ 5x slower
## Pick c1: 0 ≤ c1 ≤ n random
## i <- 1
## repeat n-1 times
##   repeat
##     i++
##     c[i] = nearest neighbor c[i-1]
##   until c[i] = c[i-2] ## nearest of nearest is cluster itself
##   merge c[i] and nearest neighbor c[i]
##   if i>3 i -= 3 else i <- 1
function hclust2(d::AbstractMatrix, linkage::Function)
    T = eltype(linkage(d, 1:0, 1:0))
    n = size(d,1)                       # number of datapoints
    mleft = Vector{Int}()               # id of left merged subtree
    mright = Vector{Int}()              # id of right merged subtree
    h = Vector{T}()                     # tree height
    cl = [[x] for x in 1:n]             # elements of active trees
    trees = collect(-(1:n))             # ids of active trees
    NN = [1]                            # nearest neighbors chain of tree indices, init by random tree index
    while length(trees) > 1
        # search for a pair of closest clusters,
        # they would be mutual nearest neighbors on top of the NN stack
        NNmindist = typemax(T)
        while true
            NNtop = NN[end]
            els_top = cl[NNtop]
            ## find NNnext: the nearest neighbor of NNtop and the next stack top
            NNnext = NNtop > 1 ? 1 : 2
            NNmindist = linkage(d, els_top, cl[NNnext])
            for k in (NNnext+1):length(cl) if k != NNtop
                dist = linkage(d, cl[k], els_top)
                if dist < NNmindist
                    NNmindist = dist
                    NNnext = k
                end
            end end
            if length(NN) > 1 && NNnext == NN[end-1] # NNnext==NN[end-1] and NNtop=NN[end] are mutual n.neighbors
                break
            else
                push!(NN, NNnext)   # grow the chain
            end
        end
        ## merge NN[end] and its nearest neighbor, i.e., NN[end-1]
        NNlo = pop!(NN)
        NNhi = pop!(NN)
        if NNlo > NNhi
             NNlo, NNhi = NNhi, NNlo
        end
        ## record the merge
        push!(mright, trees[NNlo])
        push!(mleft, trees[NNhi])
        push!(h, NNmindist)
        trees[NNlo] = length(h) # assign new id for the resulting tree
        ## merge the elements of NNlo and NNhi
        append!(cl[NNlo], cl[NNhi])
        empty!(cl[NNhi])
        ## replace any nearest neighbor referring to the last_tree with NNhi
        last_tree = length(trees)
        if NNhi < last_tree
            cl[NNhi] = cl[last_tree]
            trees[NNhi] = trees[last_tree]
            for k in eachindex(NN)
                if NN[k] == last_tree
                    NN[k] = NNhi
                end
            end
        end
        pop!(trees)
        pop!(cl)
        isempty(NN) && push!(NN, 1) # restart NN chain
    end
    ## fix order for presenting result
    o = rorder!(mleft, mright, h)
    hcat(mleft[o], mright[o]), h[o]
end

## this calls the routine that gives the correct answer, fastest
## linkage names are inspired by R's hclust
function hclust(d::AbstractMatrix; linkage::Symbol = :single,
                uplo::Union{Symbol, Nothing} = nothing)
    if uplo !== nothing
        sd = Symmetric(d, uplo) # use upper/lower part of d
    else
        assertdistancematrix(d)
        sd = d
    end
    if linkage == :single
        h = hclust_minimum(sd)
    elseif linkage == :complete
        h = hclust2(sd, slicemaximum)
    elseif linkage == :average
        h = hclust2(sd, slicemean)
    else
        throw(ArgumentError("Unsupported cluster linkage $linkage"))
    end

    # compute an ordering of the leaves
    inds = Any[]
    merge = h[1]
    for i in 1:size(merge)[1]
        inds1 = merge[i,1] < 0 ? -merge[i,1] : inds[merge[i,1]]
        inds2 = merge[i,2] < 0 ? -merge[i,2] : inds[merge[i,2]]
        push!(inds, [inds1; inds2])
    end

    Hclust(h..., inds[end], linkage)
end

@deprecate hclust(d, method::Symbol, uplo::Union{Symbol, Nothing} = nothing) hclust(d, linkage=method, uplo=uplo)

## cut a tree at height `h' or to `k' clusters
function cutree(hclu::Hclust; k::Int=1, h::Real=height(hclu))
    clusters = Vector{Int}[]
    n = nnodes(hclu)
    nodes = [[i] for i=1:n]
    cutm = n - k    # how many tree merges to do
    i = 1
    while i ≤ cutm && hclu.heights[i] ≤ h
        both = view(hclu.merges, i, :)
        newclu = Int[]
        for x in both
            if x < 0
                push!(newclu, -x)
                nodes[-x] = []
            else
                append!(newclu, clusters[x])
                clusters[x] = []
            end
        end
        push!(clusters, newclu)
        i += 1
    end
    all = filter!(!isempty, vcat(clusters, nodes))
    ## convert to a single array of cluster indices
    res = fill(0, n)
    for (i, cl) in enumerate(all)
        res[cl] .= i
    end
    return res
end

## some diagnostic functions, not exported
function printupper(d::Matrix)
    n = size(d,1)
    for i in 1:(n-1)
        print(" " ^ ((i-1) * 6))
        for j in (i+1):n
            print(@sprintf("%5.2f ", d[i,j]))
        end
        println()
    end
end
