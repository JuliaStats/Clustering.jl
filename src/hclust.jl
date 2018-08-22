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

# the sequence of tree merges
struct HclustMerges{T<:Real}
    nnodes::Int         # number of datapoints (leaf nodes)
    heights::Vector{T}  # tree height
    mleft::Vector{Int}  # ID of the left subtree merged
    mright::Vector{Int} # ID of the right subtree merged

    function HclustMerges{T}(nnodes::Integer) where {T<:Real}
        ntrees = max(nnodes-1, 0)
        new{T}(nnodes, sizehint!(T[], ntrees),
               sizehint!(Int[], ntrees), sizehint!(Int[], ntrees))
    end
end

nmerges(hmer::HclustMerges) = length(hmer.heights)
nnodes(hmer::HclustMerges) = hmer.nnodes

# merges i-th and j-th subtrees into a new tree of height h and returns its index
function push_merge!(hmer::HclustMerges{T}, i::Integer, j::Integer, h::T) where T<:Real
    if !_isrordered(i, j)
        i, j = j, i
    end
    push!(hmer.mleft, i)
    push!(hmer.mright, j)
    push!(hmer.heights, h)
    return nmerges(hmer)
end

#= utilities for working with the vector of clusters =#

cluster_size(cl::AbstractVector{Vector{Int}}, i::Integer) =
    return i > 0 ? length(cl[i]) : #= leaf node =# 1

# indices of nodes assigned to i-th cluster
# if i-th cluster is a leaf node (i < 0), return leafcluster setting its contents to [-i]
function cluster_elems(clusters::AbstractVector{Vector{Int}}, i::Integer,
                       leafcluster::AbstractVector{Int})
    if i > 0
        return clusters[i]
    else # i < 0 means it's a leaf node
        @assert length(leafcluster) == 1
        @inbounds leafcluster[1] = -i
        return leafcluster
    end
end

# merges i-th and j-th clusters and adds the result to the end of the `cl` list;
# i-th and j-th clusters are deactivated (emptied or replaced by `noels` vector)
# if either i or j are negative, the corresponding cluster is a leaf node (-i or -j, resp.)
function merge_clusters!(cl::AbstractVector{Vector{Int}},
                         i::Integer, j::Integer,
                         noels::Vector{Int} = Int[])
    if j < 0 # negative == cluster is a leaf node (-j)
        newclu = i < 0 ? [-i, -j] : push!(cl[i], -j)
    else
        clj = cl[j]
        if i < 0
            newclu = pushfirst!(clj, -i)
            cl[j] = noels
        else
            newclu = append!(cl[i], clj)
            empty!(clj) # not used anymore
        end
    end
    if i > 0
        cl[i] = noels # not used anymore
    end
    return push!(cl, newclu)
end

# compute resulting leaves (original datapoints) permutation
# given a sequence of tree nodes merges
function hclust_perm(hmer::HclustMerges)
    n = nmerges(hmer)
    perm = fill(1, nnodes(hmer)) # resulting permutation
    clusters = Vector{Int}[]     # clusters elements
    onel = [0]                   # placeholder for the elements of a leaf node
    noels = Int[]                # placeholder for empty decativated trees
    for i in 1:n
        ml = hmer.mleft[i]
        mr = hmer.mright[i]
        # elements in the right subtree are moved length(ml) positions from the start
        nl = cluster_size(clusters, ml)
        @inbounds for i in cluster_elems(clusters, mr, onel)
            perm[i] += nl
        end
        merge_clusters!(clusters, ml, mr, noels)
    end
    return perm
end

# convert HclustMerges to Hclust
function Hclust(hmer::HclustMerges, method::Symbol)
    Hclust(hcat(hmer.mleft, hmer.mright), hmer.heights,
           invperm(hclust_perm(hmer)), method)
end

## This seems to work like R's implementation, but it is extremely inefficient
## This probably scales O(n^3) or worse. We can use it to check correctness
function hclust_n3(d::AbstractMatrix, linkage::Function)
    assertdistancematrix(d)
    T = eltype(method(d, 1:0, 1:0))
    hmer = HclustMerges{T}(size(d, 1))
    n = nnodes(hmer)
    node2cl = collect(-(1:n))   # datapoint to tree attribution, initially all leaves
    cols = fill(false, n)
    rows = fill(false, n)
    mask = falses(n)
    while nmerges(hmer) + 1 < n
        # find the closest pair of trees
        NNmindist = typemax(T)
        NNi = NNj = 0           # indices of nearest neighbors clusters
        cl = unique(node2cl)    # active tree ids
        for j in eachindex(cl)  # loop over for lower triangular indices, i>j
            clj = cl[j]
            cols .= node2cl .== clj
            for i in (j+1):length(cl)
                cli = cl[i]
                rows .= node2cl .== cli
                dist = linkage(d, rows, cols) # very expensive
                if (NNi == 0) || (dist < NNmindist)
                    NNmindist = dist
                    NNi = cli
                    NNj = clj
                    mask .= cols .| rows
                end
            end
        end
        newtree = push_merge!(hmer, NNi, NNj, NNmindist)
        node2cl[mask] = newtree
    end
    return hmer
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
    hmer = HclustMerges{T}(size(d, 1))
    n = nnodes(hmer)
    ## For each 0 < i ≤ n compute Nearest Neighbor NN[i]
    NN = [nearest_neighbor(d, i, n)[1] for i in 1:n]
    ## the main loop
    trees = collect(-(1:n))  # indices of active trees, initialized to all leaves
    while length(trees) > 1  # O(n)
        # find a pair of nearest trees, i and j
        i = 1
        NNmindist = i < NN[i] ? d[i, NN[i]] : d[NN[i], i]
        for k in 2:length(trees) # O(n)
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
        trees[i] = push_merge!(hmer, trees[i], trees[j], NNmindist)
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
    return hmer
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
function rorder!(hmer::HclustMerges)
    o = sortperm(hmer.heights)
    io = invperm(o)
    ml = hmer.mleft
    mr = hmer.mright
    for i in eachindex(ml)
        if ml[i] > 0
            ml[i] = io[ml[i]]
        end
        if mr[i] > 0
            mr[i] = io[mr[i]]
        end
        if !_isrordered(ml[i], mr[i])
            ml[i], mr[i] = mr[i], ml[i]
        end
    end
    permute!(ml, o)
    permute!(mr, o)
    Base.permute!!(hmer.heights, o)
    return hmer
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
    hmer = HclustMerges{T}(size(d, 1))
    n = nnodes(st)          # number of datapoints
    cl = [[x] for x in 1:n] # contents of trees, initially leaves
    trees = collect(-(1:n)) # ids of active trees
    NN = [1]                # nearest neighbors chain of positions in trees/cl, init by random choice
    while length(cl) > 1
        found = false
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
        ## first, store the result
        trees[NNlo] = push_merge!(hmer, trees[NNlo], trees[NNhi], NNmindist)
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
    return rorder!(hmer)
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
        hmer = hclust_minimum(sd)
    elseif linkage == :complete
        hmer = hclust2(sd, slicemaximum)
    elseif linkage == :average
        hmer = hclust2(sd, slicemean)
    else
        throw(ArgumentError("Unsupported cluster linkage $linkage"))
    end

    Hclust(hmer, linkage)
end

@deprecate hclust(d, method::Symbol, uplo::Union{Symbol, Nothing} = nothing) hclust(d, linkage=method, uplo=uplo)

## cut a tree at height `h' or to `k' clusters
function cutree(hclu::Hclust;
                k::Union{Integer, Nothing} = nothing,
                h::Union{Real, Nothing} = nothing)
    # check k and h
    (k !== nothing || h !== nothing) ||
        throw(ArgumentError("Either `k` or `h` must be specified"))
    n = nnodes(hclu)
    m = nmerges(hclu)
    # use k and h to calculate how many merges to do before cutting
    if k !== nothing
        k >= min(1, n) || throw(ArgumentError("`k` should be greater or equal $(min(1,n))"))
        cutm = n - k
    else
        cutm = m
    end
    if h !== nothing
        # adjust cutm w.r.t h
        hix = findlast(hh -> hh ≤ h, hclu.heights)
        if hix !== nothing && hix < cutm
            cutm = hix
        end
    end
    clusters = Vector{Int}[]
    unmerged = fill(true, n) # if a node is not merged to a cluster
    i = 1
    while i ≤ cutm
        both = view(hclu.merges, i, :)
        newclu = Int[]
        for x in both
            if x < 0 # -x is a leaf node
                push!(newclu, -x)
                unmerged[-x] = false
            else # x is a cluster, merge to newclu
                append!(newclu, clusters[x])
                empty!(clusters[x])
            end
        end
        push!(clusters, newclu)
        i += 1
    end
    ## build an array of cluster indices (R's order)
    res = fill(0, n)
    # sort non-empty clusters by the minimal element index
    filter!(!isempty, clusters)
    permute!(clusters, sortperm(minimum.(clusters)))
    i = findfirst(unmerged)
    next = 1
    for clu in clusters
        cl1 = minimum(clu)
        while (i !== nothing) && (i < cl1)
            res[i] = next
            next += 1
            i = findnext(unmerged, i+1)
        end
        res[clu] .= next
        next += 1
    end
    while i !== nothing
        res[i] = next
        next += 1
        i = findnext(unmerged, i+1)
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
