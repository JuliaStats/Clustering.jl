# HDBSCAN Graph
# edge[i] is a list of edges adjacent to the i-th vertex
# the second element of HDBSCANEdge is the mutual reachability distance.
HDBSCANEdge = Tuple{Int, Float64}
struct HDBSCANGraph
    edges::Vector{Vector{HDBSCANEdge}}
    HDBSCANGraph(nv::Integer) = new([HDBSCANEdge[] for _ in 1 : nv])
end

function add_edge!(G::HDBSCANGraph, v1::Integer, v2::Integer, dist::Number)
    push!(G.edges[v1], (v2, dist))
    push!(G.edges[v2], (v1, dist))
end

Base.getindex(G::HDBSCANGraph, i::Int) = G.edges[i]

struct MSTEdge
    v1::Integer
    v2::Integer
    dist::Number
end
expand(edge::MSTEdge) = (edge.v1, edge.v2, edge.dist)
Base.isless(edge1::MSTEdge, edge2::MSTEdge) = edge1.dist < edge2.dist

"""
    HdbscanCluster(..., points, stability, ...)
- `points`: vector of points which belongs to the cluster
- `stability`: stablity of the cluster(-1 for noise clusters)

You can use `length` to know the number of pooints in the cluster. And `isnoise` funciton is also available to know whether the cluster is noise or not.
"""
mutable struct HdbscanCluster
    parent::Int
    children::Vector{Int}
    points::Vector{Int}
    λp::Vector{Float64}
    stability::Float64
    children_stability::Float64
    function HdbscanCluster(points::Union{Vector{Int}, Nothing})
        noise = points === nothing
        return new(0, Int[], noise ? Int[] : points, Float64[], noise ? -1 : 0, noise ? -1 : 0)
    end
end

Base.length(c::HdbscanCluster) = size(c.points, 1)
isnoise(c::HdbscanCluster) = c.stability == -1
isstable(c::HdbscanCluster) = c.stability != 0
function increment_stability(c::HdbscanCluster, λbirth)
    c.stability += sum(c.λp) - length(c.λp) * λbirth
end

"""
    HdbscanResult(k, minpts, clusters)
- `k`: we will define "core distance of point A" as the distance between point A and the `k` th neighbor point of point A.
- `min_cluster_size`: minimum number of points in the cluster
- `clusters`: result vector of clusters
"""
struct HdbscanResult
    clusters::Vector{HdbscanCluster}
    assignments::Vector{Int}
end

"""
    hdbscan(points::AbstractMatrix, k::Int, min_cluster_size::Int; gen_mst::Bool=true, mst=nothing)
Density-Based Clustering Based on Hierarchical Density Estimates.
This algorithm performs clustering as follows.
1. generate a minimum spanning tree
2. build a HDBSCAN hierarchy
3. extract the target cluster
4. generate the list of cluster assignment for each point
The detail is so complex it is difficult to explain the detail in here. But, if you want to know more about this algorithm, you should read [this docs](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).

# Parameters
- `points`: the d×n matrix, where each column is a d-dimensional coordinate of a point
- `k`: we will define "core distance of point A" as the distance between point A and the `k` th neighbor point of point A.
- `min_cluster_size`: minimum number of points in the cluster
- `gen_mst`: whether to generate minimum-spannig-tree or not
- `mst`: when is specified and `gen_mst` is false, new mst won't be generated
"""
function hdbscan(points::AbstractMatrix, k::Int, min_cluster_size::Int; metric=SqEuclidean())
    if min_cluster_size < 1
        throw(DomainError(min_cluster_size, "The `min_cluster_size` must be greater than or equal to 1"))
    end
    n = size(points, 2)
    dists = pairwise(metric, points; dims=2)
    #calculate core distances for each point
    core_dists = core_distances(dists, k)
    #calculate mutual reachability distance between any two points
    mrd = hdbscan_graph(core_dists, dists)
    #compute a minimum spanning tree by prim method
    mst = hdbscan_minspantree(mrd, n)
    #build a HDBSCAN hierarchy
    hierarchy = hdbscan_clusters(mst, min_cluster_size)
    #extract the target cluster
    prune_cluster!(hierarchy)
    #generate the list of cluster assignment for each point
    result = HdbscanCluster[]
    noise_points = fill(-1, n)
    for (i, j) in enumerate(hierarchy[2n-1].children)
        c = hierarchy[j]
        push!(result, c)
        for k in c.points
            noise_points[k] = 0
        end
    end
    push!(result, HdbscanCluster(Int[]))
    result[end].points = findall(x->x==-1, noise_points)
    assignments = Array{Int}(undef, length(points))
    for i in 1 : length(result)-1
        assignments[result[i].points] = i
    end
    assignments[result[end].points] .= -1
    return HdbscanResult(result, assignments)
end

# calculate the core distances of the points
function core_distances(dists::AbstractMatrix, k::Integer)
    core_dists = Array{Float64}(undef, size(dists, 1))
    for i in axes(dists, 1)
        dist = sort(dists[i, :])
        core_dists[i] = dist[k]
    end
    return core_dists
end

function hdbscan_graph(core_dists::AbstractVector, dists::AbstractMatrix)
    n = size(dists, 1)
    graph = HDBSCANGraph(div(n * (n-1), 2))
    for i in 1 : n-1
        for j in i+1 : n
            c = max(core_dists[i], core_dists[j], dists[i, j])
            add_edge!(graph, i, j, c)
        end
    end
    return graph
end

function hdbscan_minspantree(graph::HDBSCANGraph, n::Integer)
    function heapput!(h, v)
        idx = searchsortedlast(h, v, rev=true)
        insert!(h, (idx != 0) ? idx : 1, v)
    end

    minspantree = Vector{MSTEdge}(undef, n-1)
    
    marked = falses(n)
    nmarked = 1
    marked[1] = true
    
    h = MSTEdge[]
    
    for (i, c) in graph[1]
        heapput!(h, MSTEdge(1, i, c))
    end
    
    while nmarked < n
        i, j, c = expand(pop!(h))
        
        marked[j] && continue
        minspantree[nmarked] = MSTEdge(i, j, c)
        marked[j] = true
        nmarked += 1
        
        for (k, c) in graph[j]
            marked[k] && continue
            heapput!(h, MSTEdge(j, k, c))
        end
    end
    return minspantree
end

function hdbscan_clusters(mst::AbstractVector{MSTEdge}, min_size::Integer)
    n = length(mst) + 1
    cost = 0
    uf = UnionFind(n)
    clusters = [HdbscanCluster(min_size > 1 ? Int[i] : Int[]) for i in 1:n]
    sort!(mst)
    
    for i in 1 : n-1
        j, k, c = expand(mst[i])
        cost += c
        λ = 1 / cost
        #child clusters
        c1 = group(uf, j)
        c2 = group(uf, k)
        #reference to the parent cluster
        println(c1, c2, n+i)
        clusters[c1].parent = clusters[c2].parent = n+i
        nc1, nc2 = isnoise(clusters[c1]), isnoise(clusters[c2])
        if !(nc1 || nc2)
            #compute stability
            increment_stability(clusters[c1], λ)
            increment_stability(clusters[c2], λ)
            #unite cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            push!(clusters, HdbscanCluster(points))
        elseif !(nc1 && nc2)
            if nc2 == true
                (c1, c2) = (c2, c1)
            end
            #record the lambda value
            append!(clusters[c2].λp, fill(λ, length(clusters[c1])))
            #unite cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            push!(clusters, HdbscanCluster(points))
        else
            #unite the noise cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            if length(points) < min_size
                push!(clusters, HdbscanCluster(Int[]))
            else
                push!(clusters, HdbscanCluster(points))
            end
        end
    end
    @assert length(clusters) == 2n - 1
    return clusters
end

function prune_cluster!(hierarchy::Vector{HdbscanCluster})
    for i in 1 : length(hierarchy)-1
        if isnoise(hierarchy[i])
            c = hierarchy[i]
            push!(hierarchy[c.parent].children, i)
            hierarchy[c.parent].children_stability += c.stability
        else
            c = hierarchy[i]
            if c.stability > c.children_stability
                push!(hierarchy[c.parent].children, i)
                hierarchy[c.parent].children_stability += c.stability
            else
                println(c.parent, i)
                append!(hierarchy[c.parent].children, c.children)
                hierarchy[c.parent].children_stability += c.children_stability
            end
        end
    end
end

# Below are utility functions for building hierarchical trees
heappush!(h, v) = insert!(h, searchsortedfirst(h, v), v)

# Union-Find
# structure for managing disjoint sets
# This structure tracks which sets the elements of a set belong to,
# and allows us to efficiently determine whether two elements belong to the same set.
mutable struct UnionFind
    parent:: Vector{Integer}  # parent[root] is the negative of the size
    label::Dict{Int, Int}
    next_id::Int

    function UnionFind(nodes::Integer)
        if nodes <= 0
            throw(ArgumentError("invalid argument for nodes: $nodes"))
        end

        parent = -ones(nodes)
        label = Dict([(i, i) for i in 1 : nodes])
        new(parent, label, nodes)
    end
end

# label of the set which element `x` belong to
group(uf::UnionFind, x) = uf.label[root(uf, x)]
# all elements that have the specified label
members(uf::UnionFind, x::Int) = collect(keys(filter(n->n.second == x, uf.label)))

# root of element `x`
# The root is the representative element of the set
function root(uf::UnionFind, x::Integer)
    if uf.parent[x] < 0
        return x
    else
        return uf.parent[x] = root(uf, uf.parent[x])
    end
end

# whether element `x` and `y` belong to the same set
function issame(uf::UnionFind, x::Integer, y::Integer)
    return root(uf, x) == root(uf, y)
end

function Base.size(uf::UnionFind, x::Integer)
    return -uf.parent[root(uf, x)]
end

function unite!(uf::UnionFind, x::Integer, y::Integer)
    x = root(uf, x)
    y = root(uf, y)
    if x == y
        return false
    end
    if uf.parent[x] > uf.parent[y]
        x, y = y, x
    end
    # unite smaller tree(y) to bigger one(x)
    uf.parent[x] += uf.parent[y]
    uf.parent[y] = x
    uf.next_id += 1
    uf.label[y] = uf.next_id
    for i in members(uf, group(uf, x))
        uf.label[i] = uf.next_id
    end
    return true
end