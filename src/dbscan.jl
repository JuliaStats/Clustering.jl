###########################################################
#
# This class implements the DBSCAN clustering algorithm,
# as described in:
#
#  A density-based algorithm for discovering clusters in
#  large spatial databases with noise (1996)
#  by Martin Ester, Hans-peter Kriegel, Jörg S, Xiaowei Xu
#
#


###########################################################
#
#   DBSCAN Results Container
#
###########################################################

type DBSCANResult
    point_id::Vector{Int}
    cluster_number::Int
end

###########################################################
#
#   Core implementation
#
#   - Data is provided as a distance matrix dmat.
#
###########################################################

#
# The core DBSCAN function
#

# It may be nice to replace the matrix dmat with a pointer
# to a distance function (which could be an inline lookup
# function if the user prefers.)

function DBSCAN{T}(
    dmat::Matrix{T},    # in: distance matrix (n x n)
    min_pts::Int,       # in: index of the current point
    eps::Float64)       # in: the epsilon parameter


    n::Int = size(dmat)[1]
    cluster_id = 1
    point_id = [0 for i = 1:n]


    for point = 1 : n
        if point_id[point] == 0
            if expand_cluster(dmat, point, cluster_id, eps, min_pts, point_id)
                cluster_id += 1
            end
        end
    end

    return DBSCANResult(point_id, cluster_id-1)
end

#
# region_query appends the list of neighbors with the indices
# of all the points within distance eps of the point with
# index current_p
#

function region_query{T}(
    dmat::Matrix{T},        # in: distance matrix (n x n)
    current_p::Int,         # in: index of the current point
    eps::Float64)           # in: the epsilon parameter

    neighbors = Int[]

    n::Int = size(dmat)[1]

    for i = 1 : n
        if dmat[current_p, i] <= eps
            push!(neighbors, i)
        end
    end

    return neighbors
end

#
# expand_cluster finds all points that are
# density reachable from the provided point.
#

function expand_cluster{T}(
    dmat::Matrix{T},
    point::Int,
    cluster_id::Int,
    eps::Float64,
    min_pts::Int,
    point_id::Vector{Int})

    # Find all neighbors of the initial point
    seeds = region_query(dmat, point, eps)

    if size(seeds, 1) < min_pts
        # The provided point is not a core point
        point_id[point] = -1
        return false
    else
        # Add all the seeds to the current cluster
        for i = seeds
            point_id[i] = cluster_id
        end

        # Remove the first point from the list of seeds
        shift!(seeds)

        # Process all the neighbors of each seed
        while size(seeds,1) > 0
            current_point = shift!(seeds) # Get next seed
            # Find its neighbors
            result = region_query(dmat, current_point, eps)

            # If current_point is a core point
            if size(result, 1) >= min_pts
                # Process each of its neighbors
                for i = 1:size(result, 1)
                    if result[i] <= 0      # If it's unclassified
                        if result[i] == 0  # and not noise
                            seeds.push!(i) # add to seed list
                        end
                        # Add it to the cluster
                        point_id[i] = cluster_id
                    end
                end
            end
        end
        return true
    end
end

