# HDBSCAN Graph
# edge[i] is a list of edges adjacent to the i-th vertex?
# the second element of HDBSCANEdge is the mutual reachability distance.
HDBSCANEdge = Tuple{Int, Float64}
struct HDBSCANGraph
    edges::Vector{Vector{HDBSCANEdge}}
    HDBSCANGraph(n) = new([HDBSCANEdge[] for _ in 1 : n])
end

function add_edge(G::HDBSCANGraph, v::Tuple{Int, Int, Float64})
    i, j, c = v
    push!(G.edges[i], (j, c))
    push!(G.edges[j], (i, c))
end

Base.getindex(G::HDBSCANGraph, i::Int) = G.edges[i]

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
    function HdbscanCluster(noise::Bool, points::Vector{Int})
        if noise
            return new(0, [], [], [], -1, -1)
        else
            return new(0, [], points, [], 0, 0)
        end
    end
end

Base.length(c::HdbscanCluster) = size(c.points, 1)
isnoise(c::HdbscanCluster) = c.stability == -1
hasstability(c::HdbscanCluster) = c.stability != 0
function compute_stability(c::HdbscanCluster, λbirth)
    c.stability += sum(c.λp.-λbirth)
end

"""
    HdbscanResult(k, minpts, clusters)
- `k`: we will define "core distance of point A" as the distance between point A and the `k` th neighbor point of point A.
- `min_cluster_size`: minimum number of points in the cluster
- `clusters`: result vector of clusters
"""
mutable struct HdbscanResult
    k::Int
    min_cluster_size::Int
    clusters::Vector{HdbscanCluster}
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
function hdbscan(points::AbstractMatrix, k::Int, min_cluster_size::Int; gen_mst::Bool=true, mst=nothing)
    if min_cluster_size < 1
        throw(DomainError(min_cluster_size, "The `min_cluster_size` must be greater than or equal to 1"))
    end
    n = size(points, 1)
    if gen_mst
        #calculate core distances for each point
        core_dists = core_dist(points, k)
        #calculate mutual reachability distance between any two points
        mrd = mutual_reachability_distance(core_dists, points)
        #compute a minimum spanning tree by prim method
        mst = prim(mrd, n)
    elseif mst == nothing
        throw(ArgumentError("if you set `gen_mst` to false, you must pass a minimum spanning tree as `mst`"))
    end
    #build a HDBSCAN hierarchy
    hierarchy = build_hierarchy(mst, min_cluster_size)
    #extract the target cluster
    extract_cluster!(hierarchy)
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
    push!(result, HdbscanCluster(true, Int[]))
    result[end].points = findall(x->x==-1, noise_points)
    return HdbscanResult(k, min_cluster_size, result)
end

function core_dist(points, k)
    core_dists = Array{Float64}(undef, size(points, 1))
    for i in 1 : size(points, 1)
        p = points[i:i, :]
        dists = vec(sum((@. (points - p)^2), dims=2))
        sort!(dists)
        core_dists[i] = dists[k]
    end
    return core_dists
end

function mutual_reachability_distance(core_dists, points)
    n = size(points, 1)
    graph = HDBSCANGraph(div(n * (n-1), 2))
    for i in 1 : n-1
        for j in i+1 : n
            c = max(core_dists[i], core_dists[j], sum((points[i, :]-points[j, :]).^2))
            add_edge(graph, (i, j, c))
        end
    end
    return graph
end

function prim(graph, n)
    minimum_spanning_tree = Array{Tuple{Float64, Int, Int}}(undef, n-1)
    
    marked = falses(n)
    marked_cnt = 1
    marked[1] = true
    
    h = []
    
    for (i, c) in graph[1]
        heappush!(h, (c, 1, i))
    end
    
    while marked_cnt < n
        c, i, j = popfirst!(h)
        
        marked[j] == true && continue
        minimum_spanning_tree[marked_cnt] = (c, i, j)
        marked[j] = true
        marked_cnt += 1
        
        for (k, c) in graph[j]
            marked[k] == true && continue
            heappush!(h, (c, j, k))
        end
    end
    return minimum_spanning_tree
end

function build_hierarchy(mst, min_size)
    n = length(mst) + 1
    cost = 0
    uf = UnionFind(n)
    Hierarchy = Array{HdbscanCluster}(undef, 2n-1)
    if min_size == 1
        for i in 1 : n
            Hierarchy[i] = HdbscanCluster(false, [i])
        end
    else
        for i in 1 : n
            Hierarchy[i] = HdbscanCluster(true, Int[])
        end
    end
    sort!(mst)
    
    for i in 1 : n-1
        c, j, k = mst[i]
        cost += c
        λ = 1 / cost
        #child clusters
        c1 = group(uf, j)
        c2 = group(uf, k)
        #reference to the parent cluster
        Hierarchy[c1].parent = Hierarchy[c2].parent = n+i
        nc1, nc2 = isnoise(Hierarchy[c1]), isnoise(Hierarchy[c2])
        if !(nc1 || nc2)
            #compute stability
            compute_stability(Hierarchy[c1], λ)
            compute_stability(Hierarchy[c2], λ)
            #unite cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            Hierarchy[n+i] = HdbscanCluster(false, points)
        elseif !(nc1 && nc2)
            if nc2 == true
                (c1, c2) = (c2, c1)
            end
            #record the lambda value
            append!(Hierarchy[c2].λp, fill(λ, length(Hierarchy[c1])))
            #unite cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            Hierarchy[n+i] = HdbscanCluster(false, points)
        else
            #unite the noise cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            if length(points) < min_size
                Hierarchy[n+i] = HdbscanCluster(true, Int[])
            else
                Hierarchy[n+i] = HdbscanCluster(false, points)
            end
        end
    end
    return Hierarchy
end

function extract_cluster!(hierarchy::Vector{HdbscanCluster})
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
                append!(hierarchy[c.parent].children, c.children)
                hierarchy[c.parent].children_stability += c.children_stability
            end
        end
    end
end

# Below are utility functions for building hierarchical trees
heappush!(h, v) = insert!(h, searchsortedfirst(h, v), v)

mutable struct UnionFind{T <: Integer}
    parent:: Vector{T}  # parent[root] is the negative of the size
    label::Dict{Int, Int}
    cnt::Int

    function UnionFind{T}(nodes::T) where T<:Integer
        if nodes <= 0
            throw(ArgumentError("invalid argument for nodes: $nodes"))
        end

        parent = -ones(T, nodes)
        label = Dict([(i, i) for i in 1 : nodes])
        new{T}(parent, label, nodes)
    end
end

UnionFind(nodes::Integer) = UnionFind{typeof(nodes)}(nodes)
group(uf::UnionFind, x)::Int = uf.label[root(uf, x)]
members(uf::UnionFind, x::Int) = collect(keys(filter(n->n.second == x, uf.label)))

function root(uf::UnionFind{T}, x::T)::T where T<:Integer
    if uf.parent[x] < 0
        return x
    else
        return uf.parent[x] = root(uf, uf.parent[x])
    end
end

function issame(uf::UnionFind{T}, x::T, y::T)::Bool where T<:Integer
    return root(uf, x) == root(uf, y)
end

function Base.size(uf::UnionFind{T}, x::T)::T where T<:Integer
    return -uf.parent[root(uf, x)]
end

function unite!(uf::UnionFind{T}, x::T, y::T)::Bool where T<:Integer
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
    uf.cnt += 1
    uf.label[y] = uf.cnt
    for i in members(uf, group(uf, x))
        uf.label[i] = uf.cnt
    end
    return true
end