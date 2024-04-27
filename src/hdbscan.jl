# HDBSCAN Graph edge: target vertex and mutual reachability distance.
HdbscanEdge = Tuple{Int, Float64}

# HDBSCAN Graph
struct HdbscanGraph
    adj_edges::Vector{Vector{HdbscanEdge}}

    HdbscanGraph(nv::Integer) = new([HdbscanEdge[] for _ in 1 : nv])
end

function add_edge!(G::HdbscanGraph, v1::Integer, v2::Integer, dist::Number)
    push!(G.adj_edges[v1], (v2, dist))
    push!(G.adj_edges[v2], (v1, dist))
end

struct MSTEdge
    v1::Integer
    v2::Integer
    dist::Number
end

Base.isless(edge1::MSTEdge, edge2::MSTEdge) = edge1.dist < edge2.dist

"""
    HdbscanCluster
A cluster generated by the [hdbscan](@ref) method, part of [HdbscanResult](@ref).
  - `points`: vector of points which belongs to the cluster
  - `stability`: stability of the cluster (-1 for noise clusters)

The stability represents how much the cluster is "reasonable". So, a cluster which has a bigger stability is better.
The noise cluster is determined not to belong to any cluster. So, you can ignore them.
See also: [isnoise](@ref)
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

"""
    isnoise(c::HdbscanCluster)
This function returns whether the cluster is the noise or not.
"""
isnoise(c::HdbscanCluster) = c.stability == -1
isstable(c::HdbscanCluster) = c.stability != 0
function increment_stability(c::HdbscanCluster, λbirth)
    c.stability += sum(c.λp) - length(c.λp) * λbirth
end

"""
    HdbscanResult 
Result of the [hdbscan](@ref) clustering.
- `clusters`: vector of clusters
- `assignments`: vectors of assignments for each points
"""
struct HdbscanResult
    clusters::Vector{HdbscanCluster}
    assignments::Vector{Int}
end

"""
    hdbscan(points::AbstractMatrix, ncore::Integer, min_cluster_size::Integer;
            metric=SqEuclidean())
Cluster `points` using Density-Based Clustering Based on Hierarchical Density Estimates (HDBSCAN) algorithm.
Refer to [HDBSCAN algorithm](@ref hdbscan_algorithm) description for the details on how the algorithm works.
# Parameters
- `points`: the *d*×*n* matrix, where each column is a *d*-dimensional point
- `ncore::Integer`: number of *core* neighbors of point, see [HDBSCAN algorithm](@ref hdbscan_algorithm) for a description
- `min_cluster_size::Integer`: minimum number of points in the cluster
- `metric`(defaults to Euclidean): the points distance metric to use.
"""
function hdbscan(points::AbstractMatrix, ncore::Integer, min_cluster_size::Int; metric=Euclidean())
    if min_cluster_size < 1
        throw(DomainError(min_cluster_size, "The `min_cluster_size` must be greater than or equal to 1"))
    end
    n = size(points, 2)
    dists = pairwise(metric, points; dims=2)
    # calculate core (ncore-th nearest) distance for each point
    core_dists = [partialsort(i_dists, ncore) for i_dists in eachcol(dists)]

    #calculate mutual reachability distance between any two points
    mrd = hdbscan_graph(core_dists, dists)
    #compute a minimum spanning tree by prim method
    mst = hdbscan_minspantree(mrd, n)
    #build a HDBSCAN hierarchy
    hierarchy = hdbscan_clusters(mst, min_cluster_size)
    #extract the target cluster
    prune_cluster!(hierarchy)
    #generate the list of cluster assignment for each point
    clusters = HdbscanCluster[]
    assignments = fill(0, n) # cluster index of each point
    for (i, j) in enumerate(hierarchy[2n-1].children)
        clu = hierarchy[j]
        push!(clusters, clu)
        assignments[clu.points] .= i
    end
    # add the cluster of all unassigned (noise) points
    noise_points = findall(==(0), assignments)
    isempty(noise_points) || push!(clusters, HdbscanCluster(noise_points))
    return HdbscanResult(clusters, assignments)
end

function hdbscan_graph(core_dists::AbstractVector, dists::AbstractMatrix)
    n = size(dists, 1)
    graph = HdbscanGraph(div(n * (n-1), 2))
    for (i, i_dists) in enumerate(eachcol(dists))
        i_core = core_dists[i]
        for j in i+1:n
            c = max(i_core, core_dists[j], i_dists[j])
            add_edge!(graph, i, j, c)
        end
    end
    return graph
end

function hdbscan_minspantree(graph::HdbscanGraph, n::Integer)
    function heapput!(h, v)
        idx = searchsortedlast(h, v, rev=true)
        insert!(h, (idx != 0) ? idx : 1, v)
    end

    minspantree = Vector{MSTEdge}(undef, n-1)
    
    marked = falses(n)
    nmarked = 1
    marked[1] = true
    
    h = MSTEdge[]

    for (i, c) in graph.adj_edges[1]
        heapput!(h, MSTEdge(1, i, c))
    end
    
    while nmarked < n
        i, j, c = pop!(h)

        marked[j] && continue
        minspantree[nmarked] = MSTEdge(i, j, c)
        marked[j] = true
        nmarked += 1

        for (k, c) in graph.adj_edges[j]
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
        j, k, c = mst[i]
        cost += c
        λ = 1 / cost
        #child clusters
        c1 = group(uf, j)
        c2 = group(uf, k)
        #reference to the parent cluster
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
                append!(hierarchy[c.parent].children, c.children)
                hierarchy[c.parent].children_stability += c.children_stability
            end
        end
    end
end

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