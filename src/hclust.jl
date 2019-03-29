## hclust.jl (c) 2014, 2017 David A. van Leeuwen
## Hierarchical clustering, similar to R's hclust()

## Algorithms are based upon C. F. Olson, Parallel Computing 21 (1995) 1313--1325.

"""
The output of [`hclust`](@ref), hierarchical clustering of data points.

Provides the bottom-up definition of the dendrogram as the sequence of
merges of the two lower subtrees into a higher level subtree.

This type mostly follows R's `hclust` class.

# Fields
 - `merges::Matrix{Int}`: ``N×2`` matrix encoding subtree merges:
   * each row specifies the left and right subtrees (referenced by their ``id``s)
     that are merged
   * negative subtree ``id`` denotes the leaf node and corresponds to the data
     point at position ``-id``
   * positive ``id`` denotes nontrivial subtree (the row `merges[id, :]`
     specifies its left and right subtrees)
 - `linkage::Symbol`: the name of *cluster linkage* function used to construct
   the hierarchy (see [`hclust`](@ref))
 - `heights::Vector{T}`: subtree heights, i.e. the distances between the left
    and right branches of each subtree calculated using the specified `linkage`
 - `order::Vector{Int}`: the data point indices ordered so that there are no
    intersecting branches on the *dendrogram* plot. This ordering also puts
    the points of the same cluster close together.

See also: [`hclust`](@ref).
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

# FIXME remove after deprecation period for merge/height/method
Base.propertynames(hclu::Hclust, private::Bool = false) =
    (:merges, :heights, :order, :linkage,
     #= deprecated =# :height, :labels, :merge, :method)

# FIXME remove after deprecation period for height/labels/merge/method
function Base.getproperty(hclu::Hclust, prop::Symbol)
    if prop == :height
        Base.depwarn("Hclust::height is deprecated, use Hclust::heights", Symbol("Hclust::height"))
        return getfield(hclu, :heights)
    elseif prop == :labels
        Base.depwarn("Hclust::labels is deprecated and will be removed in future versions", Symbol("Hclust::labels"))
        return 1:nnodes(hclu)
    elseif prop == :merge
        Base.depwarn("Hclust::merge is deprecated, use Hclust::merges", Symbol("Hclust::merge"))
        return getfield(hclu, :merges)
    elseif prop == :method
        Base.depwarn("Hclust::method is deprecated, use Hclust::linkage", Symbol("Hclust::method"))
        return getfield(hclu, :linkage)
    else
        return getfield(hclu, prop)
    end
end

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

# active trees of hclust algorithm
struct HclustTrees{T<:Real}
    merges::HclustMerges{T}     # history of tree merges
    id::Vector{Int}             # IDs of active trees
    cl::Vector{Vector{Int}}     # elements in the non-trivial trees
    noels::Vector{Int}          # empty placeholder for elements of deactivated trees

    HclustTrees{T}(n::Integer) where T<:Real =
        new{T}(HclustMerges{T}(n),
               collect(-(1:n)), # init with all leaves
               sizehint!(Vector{Int}[], n),
               Vector{Int}())
end

nmerges(htre::HclustTrees) = nmerges(htre.merges)
ntrees(htre::HclustTrees) = length(htre.id)
nnodes(htre::HclustTrees) = nnodes(htre.merges)

tree_size(htre::HclustTrees, i::Integer) = cluster_size(htre.cl, htre.id[i])

# ids of elements assigned to the tree with i-th index
# if i-th the is a leaf node, return leafcluster setting its contents to the id of that node
tree_elems(htre::HclustTrees, i::Integer,
           leafcluster::AbstractVector{Int}) =
    cluster_elems(htre.cl, htre.id[i], leafcluster)

# merges the trees referenced by indices i and j in htre.id into a new tree of height h;
# the i-th and j-th trees are deactived, their containers are emptied or replaced by `noels` placeholder
function merge_trees!(htre::HclustTrees, i::Integer, j::Integer, h::Real)
    # get tree ids
    ci = htre.id[i]
    cj = htre.id[j]
    cnew = push_merge!(htre.merges, ci, cj, h)
    # in the tree list, replace ci by cnew and cj by cnew the last tree
    htre.id[i] = cnew
    htre.id[j] = htre.id[end]
    pop!(htre.id)
    merge_clusters!(htre.cl, ci, cj, htre.noels)
    return htre
end

## This seems to work like R's implementation, but it is extremely inefficient
## This probably scales O(n^3) or worse. We can use it to check correctness
function hclust_n3(d::AbstractMatrix, linkage::Function)
    assertdistancematrix(d)
    T = eltype(linkage(d, 1:0, 1:0))
    htre = HclustTrees{T}(size(d, 1))
    onecol = [0]
    onerow = [0]
    while ntrees(htre) > 1
        # find the closest pair of trees mi and mj, mj < mi
        NNmindist = typemax(T)
        NNi = NNj = 0           # indices of nearest neighbors clusters
        for j in 1:ntrees(htre)
            cols = tree_elems(htre, j, onecol)
            for i in (j+1):ntrees(htre)
                rows = tree_elems(htre, i, onerow)
                dist = linkage(d, rows, cols) # very expensive
                if (NNi == 0) || (dist < NNmindist)
                    NNmindist = dist
                    NNi = i
                    NNj = j
                end
            end
        end
        merge_trees!(htre, NNj, NNi, NNmindist)
    end
    return htre.merges
end

"""
Base type for _reducible_ Lance–Williams cluster metrics.

The metric `d` is called _reducible_ if for any clusters `A`, `B` and `C` and
some `ρ > 0` s.t.
```
d(A, B) < ρ, d(A, C) > ρ, d(B, C) > ρ
```
it follows that
```
d(A∪B, C) > ρ
```

If the cluster metrics belongs to Lance-Williams family, there is an efficient
formula that defines `d(A∪B, C)` using `d(A, C)`, `d(B, C)` and `d(A, B)`.
"""
abstract type ReducibleMetric{T <: Real} end

"""
Distance between the clusters is the minimal distance between any pair of their
points.
"""
struct MinimalDistance{T} <: ReducibleMetric{T}
    MinimalDistance(d::AbstractMatrix{T}) where T<:Real = new{T}()
end

# update `metric` distance between `k`-th cluster and `i`-th cluster
# (`d[k, i]`, `k < i`) after `j`-th cluster was merged into `i`-th cluster
@inline update!(metric::MinimalDistance{T}, d::AbstractMatrix{T},
    k::Integer, i::Integer, d_ij::T, d_kj::T,
    ni::Integer, nj::Integer, nk::Integer
) where T =
    (d[k, i] > d_kj) && (d[k, i] = d_kj)

"""
Ward distance between the two clusters `A` and `B` is the amount by
which merging the two clusters into a single larger cluster `A∪B` would increase
the average squared distance of a point to its cluster centroid.
"""
struct WardDistance{T} <: ReducibleMetric{T}
    WardDistance(d::AbstractMatrix{T}) where T<:Real = new{typeof(one(T)/2)}()
end

# update `metric` distance between `k`-th cluster and `i`-th cluster
# (`d[k, i]`, `k < i`) after `j`-th cluster was merged into `i`-th cluster
@inline function update!(metric::WardDistance{T}, d::AbstractMatrix{T},
    k::Integer, i::Integer, d_ij::T, d_kj::T,
    ni::Integer, nj::Integer, nk::Integer
) where T
    nall = ni + nj + nk
    d[k, i] = ((ni + nk) * d[k, i] + (nj + nk) * d_kj - nk * d_ij) / nall
end

"""
Average distance between a pair of points from each clusters.
"""
struct AverageDistance{T} <: ReducibleMetric{T}
    AverageDistance(d::AbstractMatrix{T}) where T<:Real = new{typeof(one(T)/2)}()
end

# update `metric` distance between `k`-th cluster and `i`-th cluster
# (`d[k, i]`, `k < i`) after `j`-th cluster was merged into `i`-th cluster
@inline function update!(metric::AverageDistance{T}, d::AbstractMatrix{T},
    k::Integer, i::Integer, d_ij::T, d_kj::T,
    ni::Integer, nj::Integer, nk::Integer
) where T
    nij = ni + nj
    d[k, i] = (ni * d[k, i] + nj * d_kj) / nij
end

"""
Maximum distance between a pair of point from each clusters.
"""
struct MaximumDistance{T} <: ReducibleMetric{T}
    MaximumDistance(d::AbstractMatrix{T}) where T<:Real = new{T}()
end

# update `metric` distance between `k`-th cluster and `i`-th cluster
# (`d[k, i]`, `k < i`) after `j`-th cluster was merged into `i`-th cluster
@inline update!(metric::MaximumDistance{T}, d::AbstractMatrix{T},
    k::Integer, i::Integer, d_ij::T, d_kj::T,
    ni::Integer, nj::Integer, nk::Integer
) where T =
    (d[k, i] < d_kj) && (d[k, i] = d_kj)

# Update upper-triangular matrix `d` of cluster-cluster `metric`-based distances
# after merging cluster `j` into cluster `i` and
# moving the last cluster (`N`) into the `j`-th slot
function update_distance_after_merge!(
    d::AbstractMatrix{T},
    metric::ReducibleMetric{T},
    clu_size::Function,
    i::Integer, j::Integer, N::Integer
) where {T <: Real}
    @assert 1 <= i < j <= N <= size(d, 1) "1 ≤ i=$i < j=$j <= N=$N ≤ $(size(d, 1))"
    @inbounds d_ij = d[i, j]
    @inbounds ni = clu_size(i)
    @inbounds nj = clu_size(j)
    ## update d, split in ranges k<i, i<k<j, j<k≤newj
    for k in 1:i          # k ≤ i
        @inbounds update!(metric, d, k, i, d_ij, d[k,j], ni, nj, clu_size(k))
    end
    for k in (i+1):(j-1)  # i < k < j
        @inbounds update!(metric, d, i, k, d_ij, d[k,j], ni, nj, clu_size(k))
    end
    for k in (j+1):N      # j < k ≤ N
        @inbounds update!(metric, d, i, k, d_ij, d[j,k], ni, nj, clu_size(k))
    end
    ## move N-th row/col into j
    if j < N
        @inbounds d[j, j] = d[N, N]
        for k in 1:(j-1)         # k < j < N
            @inbounds d[k,j] = d[k,N]
        end
        for k in (j+1):(N-1)     # j < k < N
            @inbounds d[j,k] = d[k,N]
        end
    end
    return d
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
    mindist = MinimalDistance(d)
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
        last_tree = length(trees)
        update_distance_after_merge!(d, mindist, i -> 0, i, j, last_tree)
        trees[i] = push_merge!(hmer, trees[i], trees[j], NNmindist)
        # reassign the last tree to position j
        trees[j] = trees[last_tree]
        NN[j] = NN[last_tree]
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
function hclust_nn(d::AbstractMatrix, linkage::Function)
    T = eltype(linkage(d, 1:0, 1:0))
    htre = HclustTrees{T}(size(d, 1))
    onerow = [0]  # placeholder for a leaf node of cl_i
    onecol = [0]  # placeholder for a leaf node of cl_j
    NN = [1]      # nearest neighbors chain of tree indices, init by random tree index
    while ntrees(htre) > 1
        # search for a pair of closest clusters,
        # they would be mutual nearest neighbors on top of the NN stack
        NNmindist = typemax(T)
        while true
            NNtop = NN[end]
            els_top = tree_elems(htre, NNtop, onecol)
            ## find NNnext: the nearest neighbor of NNtop and the next stack top
            NNnext = NNtop > 1 ? 1 : 2
            NNmindist = linkage(d, els_top, tree_elems(htre, NNnext, onerow))
            for k in (NNnext+1):ntrees(htre) if k != NNtop
                dist = linkage(d, tree_elems(htre, k, onerow), els_top)
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
        last_tree = ntrees(htre)
        merge_trees!(htre, NNlo, NNhi, NNmindist)
        ## replace any nearest neighbor referring to the last_tree with NNhi
        if NNhi < last_tree
            for k in eachindex(NN)
                if NN[k] == last_tree
                    NN[k] = NNhi
                end
            end
        end
        isempty(NN) && push!(NN, 1) # restart NN chain
    end
    return rorder!(htre.merges)
end

## Nearest neighbor chain algorithm for reducible Lance-Williams metrics.
## In comparison to hclust_nn() maintains the upper-triangular matrix
## of cluster-cluster distances, so it requires O(N²) memory, but it's faster,
## because distance calculation is more efficient.
function hclust_nn_lw(d::AbstractMatrix, metric::ReducibleMetric{T}) where {T<:Real}
    dd = copyto!(Matrix{T}(undef, size(d)...), d)
    htre = HclustTrees{T}(size(d, 1))
    NN = [1]      # nearest neighbors chain of tree indices, init by random tree index
    while ntrees(htre) > 1
        # search for a pair of closest clusters,
        # they would be mutual nearest neighbors on top of the NN stack
        NNmindist = typemax(T)
        while true
            ## find NNnext: nearest neighbor of NN[end] (and the next stack top)
            NNnext, NNmindist = nearest_neighbor(dd, NN[end], ntrees(htre))
            @assert NNnext > 0
            if length(NN) > 1 && NNnext == NN[end-1] # NNnext==NN[end-1] and NN[end] are mutual n.neighbors
                break
            else
                push!(NN, NNnext)
            end
        end
        ## merge NN[end] and its nearest neighbor, i.e., NN[end-1]
        NNlo = pop!(NN)
        NNhi = pop!(NN)
        if NNlo > NNhi
            NNlo, NNhi = NNhi, NNlo
        end
        last_tree = ntrees(htre)
        ## update the distance matrix (while the trees are not merged yet)
        update_distance_after_merge!(dd, metric, i -> tree_size(htre, i), NNlo, NNhi, last_tree)
        merge_trees!(htre, NNlo, NNhi, NNmindist)
        ## replace any nearest neighbor referring to the last cluster with NNhi
        for k in eachindex(NN)
            if NN[k] == last_tree
                NN[k] = NNhi
            end
        end
        isempty(NN) && push!(NN, 1) # restart NN chain
    end
    return rorder!(htre.merges)
end

"""
    hclust(d::AbstractMatrix; [linkage], [uplo])

Perform hierarchical clustering using the distance matrix `d` and
the cluster `linkage` function.

Returns the dendrogram as an object of type [`Hclust`](@ref).

# Arguments
 - `d::AbstractMatrix`: the pairwise distance matrix. ``d_{ij}`` is the distance
    between ``i``-th and ``j``-th points.
 - `linkage::Symbol`: *cluster linkage* function to use. `linkage` defines how
   the distances between the data points are aggregated into the distances between
   the clusters. Naturally, it affects what clusters are merged on each
   iteration. The valid choices are:
   * `:single` (the default): use the minimum distance between any of the
     cluster members
   * `:average`: use the mean distance between any of the cluster members
   * `:complete`: use the maximum distance between any of the members
   * `:ward`: the distance is the increase of the average squared distance of
     a point to its cluster centroid after merging the two clusters
   * `:ward_presquared`: same as `:ward`, but assumes that the distances
     in `d` are already squared.
 - `uplo::Symbol` (optional): specifies whether the upper (`:U`) or the
   lower (`:L`) triangle of `d` should be used to get the distances.
   If not specified, the method expects `d` to be symmetric.
"""
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
        hmer = hclust_nn_lw(sd, MaximumDistance(sd))
    elseif linkage == :average
        hmer = hclust_nn_lw(sd, AverageDistance(sd))
    elseif linkage == :ward_presquared
        hmer = hclust_nn_lw(sd, WardDistance(sd))
    elseif linkage == :ward
        if sd === d
            sd = abs2.(sd)
        else
            sd .= abs2.(sd)
        end
        hmer = hclust_nn_lw(sd, WardDistance(sd))
        hmer.heights .= sqrt.(hmer.heights)
    else
        throw(ArgumentError("Unsupported cluster linkage $linkage"))
    end

    Hclust(hmer, linkage)
end

@deprecate hclust(d, method::Symbol, uplo::Union{Symbol, Nothing} = nothing) hclust(d, linkage=method, uplo=uplo)

"""
    cutree(hclu::Hclust; [k], [h])

Cuts the `hclu` dendrogram to produce clusters at the specified level of
granularity.

Returns the cluster assignments vector ``z`` (``z_i`` is the index of the
cluster for the ``i``-th data point).

# Arguments
 - `k::Integer` (optional) the number of desired clusters.
 - `h::Real` (optional) the height at which the tree is cut.

If both `k` and `h` are specified, it's guaranteed that the
number of clusters is not less than `k` and their height is not above `h`.

See also: [`hclust`](@ref)
"""
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
        elseif nmerges(hclu) >= 1 && first(hclu.heights) > h
            # corner case, the requested h smaller that the smallest nontrivial subtree
            cutm = 0
        end
    end
    clusters = Vector{Int}[]
    unmerged = fill(true, n) # if a node is not merged to a cluster
    noels = Int[]            # placeholder for empty deactivated trees
    i = 1
    while i ≤ cutm
        c1 = hclu.merges[i, 1]
        c2 = hclu.merges[i, 2]
        (c1 < 0) && (unmerged[-c1] = false)
        (c2 < 0) && (unmerged[-c2] = false)
        merge_clusters!(clusters, c1, c2, noels)
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
