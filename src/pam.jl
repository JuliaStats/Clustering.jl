type PAMOutput
    assignments::Vector{Int64}
    medoid_indicies::Vector{Int64}
end

function show(io::IO, results::PAMOutput)
    println(io, "Cluster Assignments:")
    println(io, results.assignments)
    println(io)
    println(io, "Cluster Medoids:")
    println(io, results.medoid_indicies)
end


##################################################################
## **Build** phase based on http://www.cs.umb.edu/cs738/pam1.pdf 
##################################################################

function update_DE!(D, E, S, U, dist)
    for i in [S, collect(U)]
        (D[i], min_index) = findmin(dist[S, i])
        (E[i], _) = findmin(dist[[S[1:min_index-1], S[min_index+1:end]], i])
    end
end

function update_SU!(D, E, S, U, dist)
    best_g = -Inf
    best_i = 0
    i=0
    for i in U
        g_i = 0.0
        for j in U # - {I}
            if i != j
                @inbounds g_i += max(D[j]-dist[j,i], 0.0)
            end
        end

        if g_i > best_g
            best_g = g_i
            best_i = i
        end
    end

    push!(S, i)
    setdiff!(U, i)
end

function pam_build!(D, E, S, U, dist, k)
    for i in 1:k-1
        update_SU!(D, E, S, U, dist)
        update_DE!(D, E, S, U, dist)
    end
end

function pam_build(dist, k)
    n = size(dist, 1)

    best_sumdist = Inf
    best_p = 0
    for p in 1:n
        sumdist = sum(sub(dist, :, p))
        if sumdist < best_sumdist
            best_sumdist = sumdist
            best_p = p
        end
    end
    S = [best_p]
    U = Set(setdiff(1:n, S)...)

    D = dist[collect(S), :][:]
    E = Inf*ones(n);

    pam_build!(D, E, S, U, dist, k)

    return sort(S)
end

##################################################################
## End build phase
##################################################################


##################################################################
## **Swap** phase based on steps 2 to 5 in
## [K-medois](http://en.wikipedia.org/wiki/K-medoids)
## wikipedia page and
## [Data Mining and Algorithms in R](http://bit.ly/1inWWWh)
## wikibook
## The version in the notes I used for the build phase is probably
## more memory-efficient and possibly faster
##################################################################

function assign_to_medoids(medoid_indeces, dist)
    n = size(dist, 1)
    medoid_assignments = (Int => Vector{Int})[]

    for i in 1:n
        if !in(i, medoid_indeces)
            min_dist = Inf
            min_medoid = 0
            for m in medoid_indeces
                if dist[i,m] < min_dist
                    min_dist = dist[i,m]
                    min_medoid = m
                end
            end
            if haskey(medoid_assignments, min_medoid)
                push!(medoid_assignments[min_medoid], i)
            else
                medoid_assignments[min_medoid] = [i]
            end
        end
    end
    return medoid_assignments
end

function medoid_score(m, neighbors, dist)
    score = 0.0
    for n in neighbors
        score += dist[m, n]
    end
    return score
end


function best_medoid(cluster, dist)
    best = 0.0
    best_score = Inf
    for (i, medoid) in enumerate(cluster)
        score = medoid_score(medoid, [cluster[1:i-1], cluster[i+1:end]], dist)
        if score < best_score
            best_score=score
            best = medoid
        end
    end
    return best
end

function update_medoids(medoids_and_neighbors, dist)
    medoids = Array(Int, length(medoids_and_neighbors))
    for (i,(m, neighbors)) in enumerate(medoids_and_neighbors)
        medoids[i] = best_medoid([m, neighbors], dist)
    end
    return sort!(medoids)
end

function pam_swap(medoids, dist)
    old_medoids = Int[]

    medoids_and_neighbors = [Int => [Int]]
    while medoids != old_medoids
        medoids_and_neighbors = assign_to_medoids(medoids, dist)
        old_medoids = medoids
        medoids = update_medoids(medoids_and_neighbors, dist)
    end

    return medoids_and_neighbors
end
##################################################################
## end swap phase
##################################################################

function pam{R <: Real}(dist::Matrix{R}, k::Int)
    @assert issym(dist)

    medoids = pam_build(dist, k)
    medoids_and_neighbors = pam_swap(medoids, dist)

    cluster_membership = zeros(Int, size(dist, 1))
    for (i, (medoid, neighbors)) in enumerate(medoids_and_neighbors)
        cluster_membership[[medoid, neighbors]] = i
        medoids[i] = medoid
    end
    PAMOutput(cluster_membership, medoids)
end
