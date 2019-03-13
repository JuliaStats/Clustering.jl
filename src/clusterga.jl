# Clustering Genetic Algorithm

# Hruschka, Eduardo & Ebecken, Nelson. (2003). A genetic algorithm for cluster analysis.
# Intell. Data Anal.. 7. 15-25. 10.3233/IDA-2003-7103. 

using Random
using Statistics
using LinearAlgebra
using Clustering

# Results Interface

struct CGAResult <: ClusteringResult
    assignments::Vector{Int}    # element-to-cluster assignments (n)
    counts::Vector{Int}         # number of samples assigned to each cluster (k)
    found_gen::Int              # First generation where the elite was found
    total_gen::Int              # Total generations the GA has been run
end

mutable struct CGAData{S, T}
    objs::AbstractVector{S}
    dist::AbstractMatrix{T}
    N::Int
    generations::Int
    curr_gen::Int
    population::Matrix{Int}
    scratch::Matrix{Int}
    p::Matrix{Int}
    np::Matrix{Int}
    elitev::Float64
    elited::Vector{Int}
    elite_gen::Int
    val::Vector{Float64}
    function CGAData{S, T}(objs::AbstractVector{S},
                           dist::AbstractMatrix{T},
                           N::Int, generations::Int) where {S, T <: Real}
        sz = size(dist)
        @assert sz[1] == sz[2] "The dist matrix should be symmetric"
        population = p  = Matrix{Int}(undef, (N, sz[1]+1))
        scratch =    np = Matrix{Int}(undef, (N, sz[1]+1))
        curr_gen = elite_gen = 0
        elited = fill(1, sz[1]+1)
        elitev = 0
        val = Vector{Float64}(undef, N)
        return new(objs, dist, N, generations, curr_gen, population,
                   scratch, p, np, elitev, elited, elite_gen, val)
    end
end

# Ensuring silhouttes function does not throw when one cluster is found
# also assigns the silhouette values to zeros to not be picked up for
# selection significantly.
function silhouettes_nothrow(assignments::AbstractVector{<:Integer},
                             dists::AbstractMatrix{T}) where {T <: Real}
    counts = fill(0, maximum(assignments))
    for a in assignments
        counts[a] += 1
    end
    k = length(counts)
    k > 1 && return silhouettes(assignments, counts, dists)
    S = typeof((one(T) + one(T))/2)
    return fill(zero(S), length(assignments))
end

function fitness(idx::Int, data::CGAData)
    nobj = size(data.population, 2) - 1
    vw = @view data.population[idx, 1:nobj]
    return fitness(vw, data)
end

function fitness(assignment::AbstractVector{Int}, data::CGAData)
    nobj = size(data.population, 2) - 1
    vw = @view assignment[1:nobj]
    return 1.0 + mean(silhouettes_nothrow(vw, data.dist))
end

function cga(objs::AbstractVector{S},
             dist::AbstractMatrix{T}=distance_matrix(objs),
             N::Int=length(objs)*20,
             generations::Int=50) where {S, T <: Real}
    data = CGAData{S, T}(objs, dist, N, generations)
    init_population!(data.population)
    selection!(data)
    nobj = size(data.population, 2) - 1

    N_4  = div(N, 4)
    N_2  = div(N, 2)
    N3_4 = div(3N, 4)
    for gen=1:generations
        data.curr_gen = gen
        for i = 1:N_4
            crossover!(i, N_2-i+1, data)
            mutation_split!(N_2+i, data)
            mutation_merge!(N3_4+i, data)
        end
        selection!(data)
        fval = fitness(1, data)
    end
    elited = data.elited
    assignments = elited[1:nobj]
    k = elited[end]
    counts = Vector{Int}(undef, k)
    for i=1:k
        counts[i] = length(collect(1:nobj)[assignments .== i])
    end
    result = CGAResult(assignments, counts, data.elite_gen, generations)
    return data, result
end

function selection!(data::CGAData)
    nobj = size(data.population, 2) - 1
    population = data.population
    N = data.N
    dist = data.dist
    val = data.val
    elitev, elitei = -Inf, -1
    for i = 1:N
        tval = fitness(i, data)
        elitev < tval && ((elitev, elitei) = (tval, i))
        val[i] = tval
    end
    if elitev - 1 > data.elitev
        data.elitev = elitev - 1
        copyto!(data.elited, (@view data.population[elitei, :]))
        data.elite_gen = data.curr_gen
    end
    for i=2:N
        val[i] += val[i-1]
    end
    p, np = data.population === data.p ? (data.p, data.np) : (data.np, data.p)
    lastv = val[end]
    copyto!((@view np[1, :]), data.elited)
    rn = rand(N)*lastv
    for i = 2:N
        idx = searchsortedfirst(val, rn[i])
        copyto!((@view np[i, :]), (@view p[idx, :]))
    end
    data.population, data.scratch = np, p
    return
end

@inline function assign_nearest_cluster!(c::AbstractVector{Int}, g::Vector{Int},
                                         objs::AbstractVector, nobj::Int)
    lg = length(g)
    cgs = Vector{Vector{Float64}}(undef, lg)
    tobjs = similar(objs, nobj)
    for j = 1:lg
        ntobjs = 0
        for i = 1:nobj
            if c[i] == g[j]
                tobjs[ntobjs+=1] = objs[i]
            end
        end
        cgs[j] = centroid(@view tobjs[1:ntobjs])
    end

    for i=1:nobj
        if c[i] == 0
            mind, mini = Inf, 0
            for j = 1:lg
                da = objs[i] - cgs[j]
                dda = dot(da, da)
                dda < mind && ((mind, mini) = (dda, j))
            end
            c[i] = g[mini]
        end
    end
    return
end

@inline function insert_not_found!(gx, gxc, val)
    found = false
    for j = 1:gxc
        if val == gx[j]
            found = true
            break
        end
    end
    !found && (gx[gxc+=1] = val)
    return gxc
end

@inline function get_xoverchild(a, c, k, l, g)
    gx  = Vector{Int}(undef, k)
    gxc = 0
    for i=1:l
        ai = a[i]
        c[i] = ai in g ? ai : 0
        c[i] == 0 && continue
        gxc = insert_not_found!(gx, gxc, c[i])
    end
    return resize!(gx, gxc)
end

@inline function pre_crossover!(a::AbstractVector{Int}, b::AbstractVector{Int},
                                c::AbstractVector{Int}, d::AbstractVector{Int},
                                g::AbstractVector{Int})
    k1, k2 = a[end], b[end]
    k = max(k1, k2)
    l = lastindex(a) - 1
    
    g1 = get_xoverchild(a, c, k, l, g)
    g2 = get_xoverchild(b, d, k, l, g1)
    
    gc,  gd  = Vector{Int}(undef, k), Vector{Int}(undef, k)
    gcn, gdn = 0, 0
    for i=1:l
        if c[i] == 0
            if !(b[i] in g1)
                c[i] = b[i]
                gcn = insert_not_found!(gc, gcn, c[i])
            end
        else
            gcn = insert_not_found!(gc, gcn, c[i])
        end
        if d[i] == 0
            if !(a[i] in g2)
                d[i] = a[i]
                gdn = insert_not_found!(gd, gdn, d[i])
            end
        else
            gdn = insert_not_found!(gd, gdn, d[i])
        end
    end
    return resize!(gc, gcn), resize!(gd, gdn)
end

@inline function crossover!(ia::Int, ib::Int, data::CGAData)
    population = data.population
    scratch    = data.scratch
    objs       = data.objs
    nobj = size(population, 2) - 1
    a = @view population[ia, :]
    b = @view population[ib, :]
    a == b && return

    c = @view scratch[ia, :]
    d = @view scratch[ib, :]
    k = a[end]
    g = Int[]
    while !(0 < length(g) <= k)
        randsubseq!(g, collect(1:k), 0.5)
    end
    gc, gd = pre_crossover!(a, b, c, d, g)
    assign_nearest_cluster!(c, gc, objs, nobj)
    assign_nearest_cluster!(d, gd, objs, nobj)
    copyto!(a, 1, c, 1, nobj)
    copyto!(b, 1, d, 1, nobj)
    normalize_assignment!(a, c, nobj)
    normalize_assignment!(b, c, nobj)
    return
end

function mutation_split!(ia::Int, data::CGAData{S, T}) where {S, T <: Real}
    population = data.population
    scratch    = data.scratch
    objs       = data.objs
    nobj = size(population, 2) - 1
    a  = @view population[ia, :]
    sa = @view scratch[ia, :]
    nc = a[end]
    sc = rand(1:nc)
    ids = Vector{Int}(undef, nobj)
    ln = 0
    for i=1:nobj
        if a[i] == sc
            ids[ln+=1] = i
        end
    end
    resize!(ids, ln)

    cg = centroid(@view objs[ids])
    d = Dict{Int, Float64}()
    
    l2 = div(ln+1, 2)
    sort!(ids, lt = (x, y) -> begin
          if !haskey(d, x)
          dx = objs[x] - cg
          ddx = dot(dx, dx)
          d[x] = ddx
          end
          if !haskey(d, y)
          dy = objs[y] - cg
          ddy = dot(dy, dy)
          d[y] = ddy
          end
          return d[x] < d[y]
          end
          )
    for ii = l2:ln
        a[ids[ii]] = (nc + 1)
    end
    normalize_assignment!(a, sa, nobj)
    return
end

function mutation_merge!(ia::Int, data::CGAData{S, T}) where {S, T<:Real}
    population = data.population
    scratch    = data.scratch
    objs       = data.objs
    nobj = size(population, 2) - 1
    a  = @view population[ia, :]
    sa = @view scratch[ia, :]
    nc = a[end]
    if nc > 2
        sc = rand(1:nc)
        cgs = Vector{Vector{Float64}}(undef, nc)
        tobjs = similar(objs, nobj)
        for j = 1:nc
            ntobjs = 0
            for i = 1:nobj
                if a[i] == j
                    tobjs[ntobjs+=1] = objs[i]
                end
            end
            cgs[j] = centroid(@view tobjs[1:ntobjs])
        end

        cg = cgs[sc]
        d = Inf
        j = sc
        for i = 1:nc
            i == sc && continue
            dcg = cgs[i] - cg
            ddcg = dot(dcg, dcg)
            d > ddcg && ((d, j) = (ddcg, i))
        end
        for i = 1:nobj
            a[i] == sc && (a[i] = j)
        end
    end
    normalize_assignment!(a, sa, nobj)
    return
end

normalize_assignment!(v) =
    normalize_assignment!(v, Vector{Int}(undef, length(v)-1), length(v)-1)

@inline function normalize_assignment!(v, d, nobj)
    fill!(d, 0)
    curr = 0
    for i = 1:nobj
        @inbounds vi = v[i]
        @inbounds d[vi] == 0 && (d[vi] = (curr += 1))
        @inbounds v[i] = d[vi]
    end
    @inbounds v[end] = curr
    return
end

function init_population!(p::Matrix)
    N, nobj = size(p)
    nobj -= 1
    curr = 2
    d = zeros(Int, nobj)
    for i = 1:N
        vw = @view p[i, :]
        rand!(vw, 1:curr)
        normalize_assignment!(vw, d, nobj)
        curr += 1
        curr > nobj && (curr = 2)
    end
    return
end

far_object(objs::AbstractVector{S}) where {T <: Real, S <: AbstractVector{T}} =
    fill(Inf, size(objs[1]))
centroid(objs::AbstractVector{S}) where {T <: Real, S <: AbstractVector{T}} =
    Vector{Float64}(mean(objs))
function distance_matrix(objs::AbstractVector{S}) where {T <: Real,
                                                         S <: AbstractVector{T}}
    l = length(objs)
    m = Matrix{Float64}(undef, (l, l))
    for i = 1:l
        for j = i:l
            v = objs[i] - objs[j]
            @inbounds m[i, j] = m[j, i] = sqrt(dot(v, v))
        end
    end
    return m
end