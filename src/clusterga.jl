# Clustering Genetic Algorithm

# Hruschka, Eduardo & Ebecken, Nelson. (2003). A genetic algorithm for cluster analysis.
# Intell. Data Anal.. 7. 15-25. 10.3233/IDA-2003-7103. 

using Random
using Statistics
using LinearAlgebra
using Clustering

# Results Interface

"""
    CGAResult <: ClusteringResult

Contains the results from the computation of [`cga`](@ref).

#### Members
  * `assignments::Vector{Int}` element-to-cluster assignments (`n`)
  * `counts::Vector{Int}`      number of samples assigned to each cluster (`k`)
  * `found_gen::Int`           first generation where the elite was found
  * `total_gen::Int`           total generations the GA has been run
"""
struct CGAResult <: ClusteringResult
    assignments::Vector{Int}    # element-to-cluster assignments (n)
    counts::Vector{Int}         # number of samples assigned to each cluster (k)
    found_gen::Int              # first generation where the elite was found
    total_gen::Int              # total generations the GA has been run
end

"""
    CGAData{S, T<:Real} 

Contains the data used in the computation of [`cga`](@ref).

The user does not need to query this object, but can use this as an opaque object. The results of the computation can be obtained from [`CGAResult`](@ref) object.

"""
mutable struct CGAData{S, T <: Real,
                       V <: AbstractVector{S},
                       M <: AbstractMatrix{T}}
    objs::V
    dist::M
    N::Int
    generations::Int
    curr_gen::Int
    population::Matrix{Int}
    scratch::Matrix{Int}
    vpopulation::Vector{SubArray}
    vscratch::Vector{SubArray}
    
    p::Matrix{Int}
    np::Matrix{Int}
    vp::Vector{SubArray}
    vnp::Vector{SubArray}
    
    elitev::Float64
    elited::Vector{Int}
    elite_gen::Int
    val::Vector{Float64}
    function CGAData{S, T, V, M}(objs::V,
                                 dist::M,
                                 N::Int,
                                 generations::Int) where {S,
                                                          V <: AbstractVector{S},
                                                          T <: Real,
                                                          M <: AbstractMatrix{T}}
        sz = size(dist)
        @assert sz[1] == sz[2] "The dist matrix should be symmetric"
        population = p  = Matrix{Int}(undef, (sz[1]+1, N))
        scratch =    np = Matrix{Int}(undef, (sz[1]+1, N))
        vpopulation = vp  = [@view  p[:, i] for i = 1:N ]
        vscratch    = vnp = [@view np[:, i] for i = 1:N ]
        curr_gen = elite_gen = 0
        elited = fill(1, sz[1]+1)
        elitev = 0
        val = Vector{Float64}(undef, N)
        return new(objs, dist, N, generations, curr_gen, population,
                   scratch, vpopulation, vscratch, p, np, vp, vnp,
                   elitev, elited, elite_gen, val)
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
    nobj = size(data.population, 1) - 1
    vw = @view data.population[1:nobj, idx]
    return fitness(vw, data)
end

function fitness(assignment::AbstractVector{Int}, data::CGAData)
    nobj = size(data.population, 1) - 1
    vw = @view assignment[1:nobj]
    return 1.0 + mean(silhouettes_nothrow(vw, data.dist))
end

"""
    cga(objects::V,
        distances::M=distance_matrix(objs),
        N::Int=length(objs)*20,
        generations::Int=50) where {S, V <: AbstractVector{S},
                                    T <: Real, M <: AbstractMatrix{T}}

Compute the optimal clustering in the data by Genetic Algorithm over the computed mean silhouettes.

  * `objects` the vector of the objects for which clusters are to be computed.
  * `distances` the distance matrix providing the pairwise distances bewtween the `objects`
  * `N` the population size for GA computation
  * `generations` number of generations the GA has to be run

The `fitness` function used is `1+mean(silhouettes())` to ensure positive values.

#### Return Values
The method returns a tuple of ([`CGAData`](@ref), [`CGAResult`](@ref))

#### References
  1. Hruschka, Eduardo & Ebecken, Nelson. (2003). A genetic algorithm for cluster analysis. Intell. Data Anal.. 7. 15-25. 10.3233/IDA-2003-7103. 
"""
function cga(objs::V,
             dist::M=distance_matrix(objs),
             N::Int=length(objs)*20,
             generations::Int=50) where {S, V <: AbstractVector{S},
                                         T <: Real, M <: AbstractMatrix{T}}
    data = CGAData{S, T, V, M}(objs, dist, N, generations)
    init_population!(data.population)
    selection!(data)
    nobj = size(data.population, 1) - 1

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
    nobj = size(data.population, 1) - 1
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
        copyto!(data.elited, data.vpopulation[elitei])
        data.elite_gen = data.curr_gen
    end
    for i=2:N
        val[i] += val[i-1]
    end
    p, np, vp, vnp = data.population === data.p ? (data.p, data.np, data.vp, data.vnp) :
                                                  (data.np, data.p, data.vnp, data.vp)
    lastv = val[end]
    copyto!(vnp[1], data.elited)
    rn = rand(N)*lastv
    for i = 2:N
        idx = searchsortedfirst(val, rn[i])
        copyto!(vnp[i], vp[idx])
    end
    data.population, data.scratch, data.vpopulation, data.vscratch = np, p, vnp, vp
    return
end

@inline function assign_nearest_cluster!(c, objs, nobj, g)
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
        cgs[j] = centroid(tobjs[1:ntobjs])
    end

    da = similar(objs[1], Float64)
    for i=1:nobj
        if c[i] == 0
            mind, mini = Inf, 0
            for j = 1:lg
                copyto!(da, cgs[j])
                da .-= objs[i]
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
    vpopulation = data.vpopulation
    vscratch    = data.vscratch
    objs       = data.objs
    dist       = data.dist
    nobj = size(data.population, 1) - 1
    a = vpopulation[ia]
    b = vpopulation[ib]
    a == b && return

    c = vscratch[ia]
    d = vscratch[ib]
    k = a[end]
    g = Int[]
    while !(0 < length(g) <= k)
        randsubseq!(g, collect(1:k), 0.5)
    end
    gc, gd = pre_crossover!(a, b, c, d, g)
    assign_nearest_cluster!(c, objs, nobj, gc)
    assign_nearest_cluster!(d, objs, nobj, gd)
    copyto!(a, 1, c, 1, nobj)
    copyto!(b, 1, d, 1, nobj)
    normalize_assignment!(a, c, nobj)
    normalize_assignment!(b, c, nobj)
    return
end

function mutation_split!(ia::Int, data::CGAData{S, T}) where {S, T <: Real}
    vpopulation = data.vpopulation
    vscratch    = data.vscratch
    objs       = data.objs
    nobj = size(data.population, 1) - 1
    a  = vpopulation[ia]
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
    sa = vscratch[ia]
    normalize_assignment!(a, sa, nobj)
    return
end

function mutation_merge!(ia::Int, data::CGAData{S, T}) where {S, T<:Real}
    vpopulation = data.vpopulation
    vscratch    = data.vscratch
    objs       = data.objs
    dist       = data.dist
    nobj = size(data.population, 1) - 1
    a  = vpopulation[ia]
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
        sa = vscratch[ia]
        normalize_assignment!(a, sa, nobj)
    end
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
    nobj, N = size(p)
    nobj -= 1
    curr = 2
    d = zeros(Int, nobj)
    for i = 1:N
        vw = @view p[:, i]
        rand!(vw, 1:curr)
        normalize_assignment!(vw, d, nobj)
        curr += 1
        curr > nobj && (curr = 2)
    end
    return
end

far_object(objs::AbstractVector{S}) where {T <: Real, S <: AbstractVector{T}} =
    fill(Inf, size(objs[1]))

function centroid(objs::AbstractVector{S}) where {T <: Real, S <: AbstractVector{T}}
    l = length(objs)
    result = fill(0.0, length(objs[1]))
    for i=1:lastindex(objs)
        result .+= objs[i]
    end
    return result /= l
end

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
