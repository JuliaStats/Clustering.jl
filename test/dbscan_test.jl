#############################################################
#
#  This script tests the DBSCAN function and the two
#  auxiliary functions that it calls.
#

module dbscan_tester

include("../src/dbscan.jl")

function MakeTestMatrix()
    n = 125     # Number of points in the point cloud
    mp = 2      # MinPts parameter for DBSCAN
    eps = 3.1   # epsilon parameter for DBSCAN

    # An array defining the centers of the two
    # Gaussian blobs
    centers = transpose(reshape([0,0,10,10],2,2))

    x = zeros(n,3)          # An empty array for the points

    m = 1
    for i = 1 : 5
        for j = 1 : 5
            for k = 1 : 5
                x[m,1] = float(i*3)
                x[m,2] = float(j*2)
                x[m,3] = float(k)
                m += 1
            end
        end
    end

    # Turn the set of points into a distance matrix that will
    # be used by DBSCAN.
    dmat = zeros(n,n)
    for i = 1 : n
        for j = i+1:n
            d = ((x[i,1] - x[j,1])^2 + 
                 (x[i,2] - x[j,2])^2 + 
                 (x[i,3] - x[j,3])^2)^.5
            dmat[i,j] = d
            dmat[j,i] = d
        end
    end
    return dmat
end

function testRegionQuery()
    passed = true

    # Check that region_query find the correct neighbors
    # for epsilon = 1,2,3

    if !(region_query(test_matrix, 3, 1.0) == [2,3,4])
        passed = false
    end

    if !(region_query(test_matrix, 3, 2.0) == [1,2,3,4,5,8])
        passed = false
    end

    if !(region_query(test_matrix, 3, 3.0) == [1,2,3,4,5,6,7,8,9,10,28])
        passed = false
    end

    return passed
end

function testExpandCluster()
    passed = true

    # Make the list of cluster ids for expand_cluster
    # to modify
    r = [0 for i = 1:125]

    expand_cluster(test_matrix, 1, 1, 2.2, 3, r)

    # Make a list of points in the new cluster
    cluster = Int[]

    for i = 1:125
        if (r[i] == 1)
            push!(cluster, i)
        end
    end

    # Should be the first 25 points
    if (cluster != [i for i = 1:25])
        passed = false
    end
        
    return passed               
end

function testDB()
    passed = true

    # Run DBSCAN.
    r = DBSCAN(test_matrix, 2, 2.1)

    # Check that the number of clusters is correct
    if (r.cluster_number != 5)
        passed = false
    else
        # Check that each cluster contains a block of
        # 25 consecutive integers
        for i = 1 : r.cluster_number
            cluster = Int[]

            # Make a list of the points in the cluster
            for j = 1 : size(r.point_id)[1]
                if r.point_id[j] == i
                    push!(cluster, j)
                end
            end
            
            # Check that this list is correct 
            if (cluster != [n for n = ((i*25)-24):(i*25)])
                passed = false
            end
        end
    end

    return passed
end

end # end module

test_matrix = MakeTestMatrix()

println ("region_query test: ", testRegionQuery())
println ("expand_cluster test: ", testExpandCluster())
println ("dbscan test: ", testDB())


