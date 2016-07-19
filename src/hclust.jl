## hclust.jl (c) 2014 David A. van Leeuwen
## Hierarchical clustering, similar to R's hclust()

## Algorithms are based upon C. F. Olson, Parallel Computing 21 (1995) 1313--1325.

## This is also in types.jl, but that is not read...
## Mostly following R's hclust class
type Hclust{T<:Real}
    merge::Matrix{Int}
    height::Vector{T}
    order::Vector{Int}
    labels::Vector
    method::Symbol
end

function assertdistancematrix(d::AbstractMatrix)
    nr, nc = size(d)
    nr == nc || throw(DimensionMismatch("Distance matrix should be square."))
    issymmetric(d) || error("Distance matrix should be symmetric.")
end

## This seems to work like R's implementation, but it is extremely inefficient
## This probably scales O(n^3) or worse. We can use it to check correctness
function hclust_n3{T<:Real}(d::AbstractMatrix{T}, method::Function)
    assertdistancematrix(d)
    mr = Int[]                  # min row
    mc = Int[]                  # min col
    h = T[]                     # height
    nc = size(d,1)              # number of clusters
    cl = -[1:nc]                # segment to cluster attribution, initially negative
    next = 1                    # next cluster label
    while next < nc
        mindist = Inf
        mi = mj = 0
        cli = unique(cl)
        mask = BitVector(nc)
        for j in 1:length(cli)           # loop over for lower triangular indices, i>j
            cols = cl .== cli[j]
            for i in (j+1):length(cli)
                rows = cl.==cli[i]
                distance = method(d[rows,cols]) # very expensive
                if distance < mindist
                    mindist = distance
                    mi = cli[i]
                    mj = cli[j]
                    mask = cols | rows
                end
            end
        end
        ## simulate R's order
        if mi < 0 && mj < 0 && mi > mj ||
            mi > 0 && mj > 0 && mi < mj ||
            mi < 0 && mj > 0
            push!(mr, mi)
            push!(mc, mj)
        else
            push!(mr, mj)
            push!(mc, mi)
        end
        push!(h, mindist)
        cl[mask] = next
        next += 1
    end
    hcat(mr, mc), h
end

## Efficient single link algorithm, according to Olson, O(n^2), fig 2.
## Verified against R's implementation, correct, and about 2.5 x faster
## For each i < j compute D(i,j) (this is already given)
## For each 0 < i <= n compute Nearest Neighbor N(i)
## Repeat n-1 times
##   find i,j that minimize D(i,j)
##   merge clusters i and j
##   update D(i,j) and N(i) accordingly
function hclust_minimum{T<:Real}(ds::Symmetric{T})
    ## For each i < j compute d[i,j] (this is already given)
    d = full(ds)                #  we need a local copy
    nc = size(d,1)
    mr = Array(Int, nc-1)       # min row
    mc = Array(Int, nc-1)       # min col
    h = Array(T, nc-1)          # height
    merges = -collect(1:nc)
    next = 1
    ## For each 0 < i <= n compute Nearest Neighbor N[i]
    N = zeros(Int, nc)
    for k = 1:nc
        mindist = Inf
        mk = 0
        for i = 1:(k-1)
            if d[i,k] < mindist
                mindist = d[i,k]
                mk = i
            end
        end
        for j = (k+1):nc
            if d[k,j] < mindist
                mindist = d[k,j]
                mk = j
            end
        end
        N[k] = mk
    end
    ## the main loop
    while nc > 1                # O(n)
        mindist = d[1,N[1]]
        i = 1
        for k in 2:nc           # O(n)
            if k < N[k]
                distance = d[k,N[k]]
            else
                distance = d[N[k],k]
            end
            if distance < mindist
                mindist = distance
                i = k
            end
        end
        j = N[i]
        if i > j
            i, j = j, i     # make sure i < j
        end
        ## update result, compatible to R's order.  It must be possible to do this simpler than this...
        if merges[i] < 0 && merges[j] < 0 && merges[i] > merges[j] ||
            merges[i] > 0 && merges[j] > 0 && merges[i] < merges[j] ||
            merges[i] < 0 && merges[j] > 0
            mr[next] = merges[i]
            mc[next] = merges[j]
        else
            mr[next] = merges[j]
            mc[next] = merges[i]
        end
        h[next] = mindist
        merges[i] = next
        merges[j] = merges[nc]
        ## update d, split in ranges k<i, i<k<j, j<k<=nc
        for k = 1:(i-1)         # k < i
            if d[k,i] > d[k,j]
                d[k,i] = d[k,j]
            end
        end
        for k = (i+1):(j-1)     # i < k < j
            if d[i,k] > d[k,j]
                d[i,k] = d[k,j]
            end
        end
        for k = (j+1):nc        # j < k <= nc
            if d[i,k] > d[j,k]
                d[i,k] = d[j,k]
            end
        end
        ## move the last row/col into j
        for k=1:(j-1)           # k==nc,
            d[k,j] = d[k,nc]
        end
        for k=(j+1):(nc-1)
            d[j,k] = d[k,nc]
        end
        ## update N[k], k !in (i,j)
        for k=1:nc
            if N[k] == j       # update nearest neigbors != i
                N[k] = i
            elseif N[k] == nc
                N[k] = j
            end
        end
        N[j] = N[nc]            # update N[j]
        ## update nc, next
        nc -= 1
        next += 1
        ## finally we need to update N[i], because it was nearest to j
        mindist = Inf
        mk = 0
        for k=1:(i-1)
            if d[k,i] < mindist
                mindist = d[k,i]
                mk = k
            end
        end
        for k = (i+1):nc
            if d[i,k] < mindist
                mindist = d[i,k]
                mk = k
            end
        end
        N[i] = mk
#        for n in N[1:nc] print(n, " ") end; println()
    end
    return hcat(mr, mc), h
end


## functions to compute maximum, minimum, mean for just a slice of an array

function slicemaximum{T<:Real}(d::AbstractMatrix{T}, cl1::Vector{Int}, cl2::Vector{Int})
    maxdist = -Inf
    for i in cl1 for j in cl2
        if d[i,j] > maxdist
            maxdist = d[i,j]
        end
    end end
    maxdist
end

function sliceminimum{T<:Real}(d::AbstractMatrix{T}, cl1::Vector{Int}, cl2::Vector{Int})
    mindist = Inf
    for i in cl1 for j in cl2
        if d[i,j] < mindist
            mindist = d[i,j]
        end
    end end
    mindist
end

function slicemean{T<:Real}(d::AbstractMatrix{T}, cl1::Vector{Int}, cl2::Vector{Int})
    s = zero(T)
    for i in cl1 for j in cl2
        s += d[i,j]
    end end
    s / (length(cl1)*length(cl2))
end

## This reorders the pairs to be compatible with R's hclust()
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
        ## R's order of pairs
        if ! (mr[i] < 0 && mc[i] < 0 && mr[i] > mc[i] ||
              mr[i] > 0 && mc[i] > 0 && mr[i] < mc[i] ||
              mr[i] < 0 && mc[i] > 0)
            mr[i], mc[i] = mc[i], mr[i]
        end
    end
    return o
end

## Another nearest neighbor algorithm, for reducible metrics
## From C. F. Olson, Parallel Computing 21 (1995) 1313--1325, fig 5
## Verfied against R implementation for mean and maximum, correct but ~ 5x slower
## Pick c1: 0 <= c1 <= n random
## i <- 1
## repeat n-1 times
##   repeat
##     i++
##     c[i] = nearest neigbour c[i-1]
##   until c[i] = c[i-2] ## nearest of nearest is cluster itself
##   merge c[i] and nearest neigbor c[i]
##   if i>3 i -= 3 else i <- 1
function hclust2{T<:Real}(d::Symmetric{T}, method::Function)
    nc = size(d,1)                      # number of clusters
    mr = Array(Int, nc-1)               # min row
    mc = Array(Int, nc-1)               # min col
    h = Array(T, nc-1)                  # height
    cl = [[x] for x in 1:nc]            # clusters
    merges = -collect(1:nc)
    next = 1
    i = 1
    N = Array(Int, nc+1)
    N[1] = 1                            # arbitrary choice
    while nc > 1
        found=false
        mindist = Inf
        while !found
            i += 1
            mi = 0
            mindist = Inf
            Nim1 = N[i-1]
            ## c[i] = nearest neigbour c[i-1]
            for j = 1:nc if Nim1 != j
                distance = method(d, cl[Nim1], cl[j])
                if distance < mindist
                    mindist = distance
                    mi = j
                end
            end end
            N[i] = mi           # N[i+1] is nearest neigbor to N[i]
            found = i > 2 && N[i] == N[i-2]
        end
        ## merge c[i] and nearest neigbor c[i], i.e., c[i-1]
        if N[i-1] < N[i]
            lo, high = N[i-1], N[i]
        else
            lo, high = N[i], N[i-1]
        end
        ## first, store the result
        mr[next] = merges[lo]
        mc[next] = merges[high]
        h[next] = mindist
        merges[lo] = next
        merges[high] = merges[nc]
        next += 1
        ## then perform the actual merge
        cl[lo] = vcat(cl[lo], cl[high])
        cl[high] = cl[nc]
        if i>3
            i -= 3
        else
            i = 1
        end
        ## replace any nearest neighbor referring to nc
        for k=1:i
            if N[k] == nc
                N[k] = high
            end
        end
        nc -= 1
    end
    ## fix order for presenting result
    o = rorder!(mr, mc, h)
    hcat(mr[o], mc[o]), h[o]
end

## this calls the routine that gives the correct answer, fastest
## method names are inspired by R's hclust
function hclust{T<:Real}(d::Symmetric{T}, method::Symbol)
    nc = size(d,1)
    if method == :single
        h = hclust_minimum(d)
    elseif method == :complete
        h = hclust2(d, slicemaximum)
    elseif method == :average
        h = hclust2(d, slicemean)
    else
        error("Unsupported method ", method)
    end

    # compute an ordering of the leaves
    inds = Any[]
    merge = h[1]
    for i in 1:size(merge)[1]
        inds1 = merge[i,1] < 0 ? -merge[i,1] : inds[merge[i,1]]
        inds2 = merge[i,2] < 0 ? -merge[i,2] : inds[merge[i,2]]
        push!(inds, [inds1; inds2])
    end

    ## label is just a placeholder for the moment
    Hclust(h..., inds[end], collect(1:nc), method)
end

## uplo may be Char for v0.3, Symbol for v0.4
hclust{T<:Real}(d::AbstractMatrix{T}, method::Symbol, uplo) = hclust(Symmetric(d, uplo), method)

function hclust{T<:Real}(d::AbstractMatrix{T}, method::Symbol)
    assertdistancematrix(d)
    hclust(Symmetric(d), method)
end


## cut a tree at height `h' or to `k' clusters
function cutree(hclust::Hclust; k::Int=1,
                h::Real=maximum(hclust.height))
    clusters = Vector{Int}[]
    nnodes = length(hclust.labels)
    nodes = [[i::Int] for i=1:nnodes]
    N = nnodes - k
    i = 1
    while i<=N && hclust.height[i] <= h
        both = vec(hclust.merge[i,:])
        new = Int[]
            for x in both
                if x<0
                    push!(new, -x)
                    nodes[-x] = []
                else
                    append!(new, clusters[x])
                    clusters[x] = []
                end
            end
        push!(clusters, new)
        i += 1
    end
    all = vcat(clusters, nodes)
    all = all[map(length, all) .> 0]
    ## convert to a single array of cluster indices
    res = Array(Int, nnodes)
    for (i,cl) in enumerate(all)
        res[cl] = i
    end
    res
end

## some diagnostic functions, not exported
function printupper(d::Matrix)
    n = size(d,1)
    for i = 1:(n-1)
        print(" " ^ ((i-1) * 6))
        for j = (i+1):n
            print(@sprintf("%5.2f ", d[i,j]))
        end
        println()
    end
end
