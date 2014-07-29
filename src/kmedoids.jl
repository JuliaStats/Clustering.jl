# This is an implementation of the algorithm given in Hae-Sang Park and
# Chi-Hyuck Jun, "A simple and fast algorithm for K-medoids clustering".
# Expert Systems with Applications 36 (2009) 3336–3341
# doi:10.1016/j.eswa.2008.01.039

type KmedoidsResult
    medoids::Vector{Int}		# indexes of medoids
    assignments::Vector{Int}            # cluster assignments
end

# Given a new set of medoids, cluster the points according to which medoid is
# closest.
function find_clusters{R <: FloatingPoint}(dist::Matrix{R}, medoids::Vector{Int})
    n = size(dist)[1]
    k = size(medoids)[1]
    total_dist = 0.0

    clusters = [Int[] for i = 1:k]

    for i = 1:n
        (distance, index) = findmin(dist[medoids,i])
        total_dist += distance

        push!(clusters[index], i) 
    end

    (total_dist, clusters)
end

# Given a new set of clusters, find the medoid of each cluster.
function new_medoids{R <: FloatingPoint}(dist::Matrix{R}, clusters::Vector{Vector{Int}})
    medoids::Vector{Int} = zeros(size(clusters)[1])

    for (i, cluster) in enumerate(clusters)
        dist_within_cluster = [sum([dist[i,j] for j in cluster]) for i in cluster]
        best = findmin(dist_within_cluster)[2]
        medoids[i] = cluster[best]
    end

    medoids
end

# Produce a flat array of cluster membership info
# eg [[1,3],[2],[4,5]] becomes [1,2,1,3,3]
# Needed for compatibility with the old implementation
function cluster_membership(clusters::Vector{Vector{Int}}, n::Int)
    assignments = zeros(n)
    for (i, cluster) in enumerate(clusters)
        for object in cluster
            assignments[object] = i
        end
    end
    assignments
end

# Calculate the k-medoids clustering of the points with dissimilarity matrix
# given by dist.
function kmedoids{R <: FloatingPoint}(dist::Matrix{R}, k::Int)
    kmedoids(dist, kmpp_by_costs(dist, k))
end

function kmedoids{R <: FloatingPoint}(dist::Matrix{R}, medoids::Vector{Int})
    n = size(dist)[1]
    k = size(medoids)[1]
    @assert 2 < k < n
    @assert issym(dist)

    (total_dist, clusters) = find_clusters(dist, medoids)
    old_dist = Inf

    while total_dist < old_dist
        old_dist = total_dist
        medoids = new_medoids(dist, clusters)
        (total_dist, clusters) = find_clusters(dist, medoids)
    end

    KmedoidsResult(medoids, cluster_membership(clusters, size(dist)[1]))
end
