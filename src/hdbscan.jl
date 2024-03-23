struct HDBSCANGraph
    edges::Vector{Vector{Tuple{Int, Float64}}}
    HDBSCANGraph(n) = new([Tuple{Int, Float64}[] for _ in 1 : n])
end

function add_edge(G::HDBSCANGraph, v::Tuple{Int, Int, Float64})
    i, j, c = v
    push!(G.edges[i], (j, c))
    push!(G.edges[j], (i, c))
end

Base.getindex(G::HDBSCANGraph, i::Int) = G.edges[i]

mutable struct HDBSCANCluster
    parent::Int
    children::Vector{Int}
    points::Vector{Int}
    λp::Vector{Float64}
    stability::Float64
    children_stability::Float64
    function HDBSCANCluster(noise::Bool, points::Vector{Int})
        if noise
            return new(0, [], [], [], -1, -1)
        else
            return new(0, [], points, [], 0, 0)
        end
    end
end

Base.length(c::HDBSCANCluster) = size(c.points, 1)
join(c1::HDBSCANCluster, c2::HDBSCANCluster, id) = HDBSCANCluster(nothing, vcat(c1.points, c2.points), id, 0)
isnoise(c::HDBSCANCluster) = c.stability == -1
hasstability(c::HDBSCANCluster) = c.stability != 0
function compute_stability(c::HDBSCANCluster, λbirth)
    c.stability += sum(c.λp.-λbirth)
end

"""
    HDBSCAN(ε, minpts)
Density-Based Clustering Based on Hierarchical Density Estimates.
This algorithm performs clustering as follows.
1. generate a minimum spanning tree
2. build a HDBSCAN hierarchy
3. extract the target cluster
4. generate the list of cluster assignment for each point
The detail is so complex it is difficult to explain the detail in here. But, if you want to know more about this algorithm, you should read [this docs](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).
"""
mutable struct HDBSCANResult
    k::Int
    min_cluster_size::Int
    labels::Union{Vector{Int}, Nothing}
    function HDBSCANResult(k::Int, min_cluster_size::Int)
        if min_cluster_size < 1
            throw(DomainError(min_cluster_size, "The `min_cluster_size` must be greater than or equal to 1"))
        end
        return new(k, min_cluster_size)
    end
end

"""
    hdbscan!(points::AbstractMatrix, k::Int, min_cluster_size::Int; gen_mst::Bool=true, mst=nothing)
# Parameters
- `points`: the d×n matrix, where each column is a d-dimensional coordinate of a point
- `k`: we will define "core distance of point A" as the distance between point A and the `k` th neighbor point of point A.
- `min_cluster_size`: minimum number of points in the cluster
- `gen_mst`: whether to generate minimum-spannig-tree or not
- `mst`: when is specified and `gen_mst` is false, new mst won't be generated
"""
function hdbscan!(points::AbstractMatrix, k::Int, min_cluster_size::Int; gen_mst::Bool=true, mst=nothing)
    model = HDBSCANResult(k, min_cluster_size)
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
    result = fill(-1, n)
    for (i, j) in enumerate(hierarchy[2n-1].children)
        c = hierarchy[j]
        for k in c.points
            result[k] = i
        end
    end
    model.labels = result
    return model
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
    Hierarchy = Array{HDBSCANCluster}(undef, 2n-1)
    if min_size == 1
        for i in 1 : n
            Hierarchy[i] = HDBSCANCluster(false, [i])
        end
    else
        for i in 1 : n
            Hierarchy[i] = HDBSCANCluster(true, Int[])
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
            Hierarchy[n+i] = HDBSCANCluster(false, points)
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
            Hierarchy[n+i] = HDBSCANCluster(false, points)
        else
            #unite the noise cluster
            unite!(uf, j, k)
            #create parent cluster
            points = members(uf, group(uf, j))
            if length(points) < min_size
                Hierarchy[n+i] = HDBSCANCluster(true, Int[])
            else
                Hierarchy[n+i] = HDBSCANCluster(false, points)
            end
        end
    end
    return Hierarchy
end

function extract_cluster!(hierarchy::Vector{HDBSCANCluster})
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
# Please note that these functions are not so sophisticated since I made them by manually converting code of numpy.
const LOG_2π = log(2π)
const newaxis = [CartesianIndex()]

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

#error function
#if absolute of `x` is smaller than 2.4, we use Taylor expansion.
#other wise, we use Continued fraction expansion
function erf(x)
    absx = abs(x)
    if absx<2.4
        c=1
        a=0
        for i in 1 : 40
            a+=(x/(2i-1)*c)
            c=-c*x^2/i
        end
        return a*2/sqrt(π)
    else
        if absx>1e50
            a = 1
        else
            y = absx*sqrt(2)
            a = 0
            for i in 40:-1:1
                a=i/(y+a)
            end
            a=1-exp(-x^2)/(y+a)*sqrt(2/π)
        end
        if x<0
            return -a
        else
            return a
        end
    end
end
erfc(x) = 1-erf(x)
norm_cdf(x) = 1/2*erfc(-x/sqrt(2))

function process_parameters(dim, mean, cov)
    if dim === nothing
        if mean === nothing
            if cov === nothing
                dim = 1
            else
                cov = convert(Array{Float64}, cov)
                if ndims(cov) < 2
                    dim = 1
                else
                    dim = size(cov, 1)
                end
            end
        else
            mean = convert(Array{Float64}, mean)
            dim = length(mean)
        end
    else
        !isa(dim, Number) && throw(DimensionMismatch("dimension of random variable must be a scalar"))
    end
    
    if mean === nothing
        mean = zeros(dim)
    end
    mean = convert(Array{Float64}, mean)
    
    if cov === nothing
        cov = [1.0]
    end
    cov = convert(Array{Float64}, cov)
    
    if dim == 1
        mean = reshape(mean, 1)
        cov = reshape(cov, 1, 1)
    end
    
    if ndims(mean) != 1 || size(mean, 1) != dim
        throw(ArgumentError("array `mean` must be vector of length $dim"))
    end
    if ndims(cov) == 0
        cov = cov * Matrix{Float64}(I, dim, dim)
    elseif ndims(cov) == 1
        cov = diag(cov)
    else
        size(cov) != (dim, dim) && throw(DimensionMismatch("array `cov` must be at most two-dimensional, but ndims(cov) = $(ndims(cov))"))
    end
    return dim, mean, cov
end

function process_quantiles(x, dim)
    x = convert(Array{Float64}, x)
    
    if ndims(x) == 0
        x = [x]
    elseif ndims(x) == 1
        if dim == 1
            x = x[:, :]
        else
            x = x[newaxis, :]
        end
    end
    return x
end

function pinv_1d(v; _eps=1e-5)
    return [(abs(x)<_eps) ? 0 : 1/x for x in v]
end

function psd_pinv_decomposed_log_pdet(mat; cond=nothing, rcond=nothing)
    u, s = eigvecs(mat), eigvals(mat)
    
    if rcond !== nothing
        cond = rcond
    end
    if cond === nothing || cond == -1
        cond = 1e6 * Base.eps()
    end
    _eps = cond * maximum(abs.(s))
    
    if minimum(s) < -_eps
        throw(ArgumentError("the covariance matrix must be positive semidefinite"))
    end
    s_pinv = pinv_1d(s, _eps=_eps)
    U = u .* sqrt.(s_pinv')
    log_pdet = sum(log.(s[findall(s.>_eps)]))

    return U, log_pdet
end

function squeeze_output(out)
    if length(out) == 1
        out = Float64(out...)
    else
        out = vec(out)
    end
    return out
end

function _logpdf(x, mean, prec_U, log_det_cov)
    dim = size(x, ndims(x))
    dev = x - mean'
    tmp = (dev * prec_U).^2
    maha = sum(tmp, dims=ndims(tmp))
    maha = dropdims(maha, dims=ndims(tmp))
    return -0.5 * ((dim*LOG_2π+log_det_cov).+maha)
end

function pdf(x, mean, cov)
    dim, mean, cov = process_parameters(nothing, mean, cov)
    x = process_quantiles(x, dim)
    prec_U, log_det_cov = psd_pinv_decomposed_log_pdet(cov)
    out = exp.(_logpdf(x, mean, prec_U, log_det_cov))
    return squeeze_output(out)
end

logpdf(x, mean, cov) = log.(pdf(x, mean, cov))