import Clustering

n = 300     # Number of points in the point cloud
mp = 5      # MinPts parameter for DBSCAN
eps = 2.0   # epsilon parameter for DBSCAN

# An array defining the centers of the two
# Gaussian blobs
centers = transpose(reshape([0,0,10,10],2,2))

x = zeros(n,2)          # An empty array for the points
m = size(centers)[1]    # The number of blob centers

# Create the point cloud
for i = 1 : n
    # Randomly select a new point by choosing coordinates
    # and a random number. If the random number is less
    # than the Gaussian function of the distance from the
    # new point to the desired center, then add the new
    # point. Otherwise, try again.
    good = 0
    while good == 0
        a = 2 * rand() - 1
        b = 2 * rand() - 1
        p = rand()
        if e^-(a^2 + b^2)^.5 >= p
            # The index i%m+1 cycles through the different
            # blob centers.
            x[i,1] = a + offset[i%m+1,1]
            x[i,2] = b + offset[i%m+1,2]
            good = 1
        end
    end
end

# Turn the set of points into a distance matrix that will
# be used by DBSCAN.
dmat = zeros(n,n)
for i = 1 : n
    for j = i+1:n
        d = ((x[i,1] - x[j,1])^2 + (x[i,2] - x[j,2])^2)^.5
        dmat[i,j] = d
        dmat[j,i] = d
    end
end

# Run DBSCAN.
r = DB.DBSCAN(dmat, mp, eps)


# Report the cluster information.
println("Number of clusters: ", r.cluster_number)

for i = 1 : r.cluster_number
    total = 0
    for j = 1 : size(r.point_id)[1]
        if r.point_id[j] == i
            total += 1
        end
    end
    println("Cluster ", i, " has ", total, " points.")
end

