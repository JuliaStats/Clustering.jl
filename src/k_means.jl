# Compute RSS, which is the squared Euclidean distance between every point
# and its assigned center
function rss{S <: Real, T <: Real}(x::Matrix{S},
                                   centers::Matrix{T},
                                   assignments::Vector{Int},
                                   n::Integer,
                                   p::Integer)
    residuals = 0.0

    for i = 1:n
        for j in 1:p
            residuals += (x[i, j] - centers[assignments[i], j])^2
        end
    end

    return residuals
end

# Find shortest distance from x to any of n_centers existing clusters
function dsquared{S <: Real, T <: Real}(x::Matrix{S},
                                        centers::Matrix{T},
                                        n_centers::Integer,
                                        p::Integer)
    m = Inf

    for i in 1:n_centers
        dst = 0.0
        for j in 1:p
            dst += (centers[i, j] - x[1, j])^2
        end
        if dst < m
            m = dst
        end
    end

    return m
end

# The k-means rule
function k_means_initialization!{S <: Real, T <: Real}(x::Matrix{S},
                                                       centers::Matrix{T},
                                                       n::Integer,
                                                       p::Integer,
                                                       k::Integer)
    for cluster in 1:k
        centers[cluster, :] = x[rand(1:n), :]
    end
end

# The k-means++ rule
function k_meanspp_initialization!{S <: Real, T <: Real}(x::Matrix{S},
                                                         centers::Matrix{T},
                                                         n::Integer,
                                                         p::Integer,
                                                         k::Integer)
    # Pre-allocate variables
    dsquareds = Array(Float64, n)
    dstr = Categorical(n)

    # Choose an initial cluster center uniformly at random from data points
    c1 = rand(1:n)
    centers[1, :] = x[c1, :]

    for cluster in 2:k
        s = 0.0
        for i in 1:n
            Dsq = dsquared(x[i, :], centers, cluster - 1, p)
            dsquareds[i] = Dsq
            s += Dsq
        end
        dstr.prob = dsquareds / s
        selected = rand(dstr)
        centers[cluster, :] = x[selected, :]
    end
end

# Update cluster assignments
function update_assignments!{S <: Real, T <: Real}(x::Matrix{S},
                                                   centers::Matrix{T},
                                                   distances::Vector{Float64},
                                                   assignments::Vector{Int},
                                                   n::Integer,
                                                   p::Integer,
                                                   k::Integer)
    for i = 1:n
        for cluster in 1:k
            d = 0.0
            for j in 1:p
                d += (x[i, j] - centers[cluster, j])^2
            end
            distances[cluster] = d
        end

        assignments[i] = indmin(distances)
    end
end

# Update cluster centers
function update_centers!(x::Matrix,
                         centers::Matrix,
                         assignments::Vector{Int},
                         n::Integer,
                         p::Integer,
                         k::Integer)
    for cluster = 1:k
        for j in 1:p
            centers[cluster, j] = 0.0
        end

        n_obs = 0
        for i in 1:n
            if assignments[i] == cluster
                n_obs += 1
                for j in 1:p
                    centers[cluster, j] += x[i, j]
                end
            end
        end

        if n_obs > 0
            for j in 1:p
                centers[cluster, j] /= n_obs
            end
        end
    end
end

function k_means{T <: Real}(x::Matrix{T}, k::Integer, opts::Options)
    # Set default options
    @defaults opts tolerance => 10e-8
    @defaults opts max_iter => 1_000
    @defaults opts initialize_centers! => k_meanspp_initialization!

    # Keep track of the number of data points and their dimensionality
    n, p = size(x)

    # Run until convergence or until the number of iterations is too high
    converged, iter = false, 0

    # Use current_rss and previous_rss to measure convergence
    previous_rss, current_rss = Inf, Inf

    # Pre-allocate arrays that are used repeatedly
    centers = Array(Float64, k, p)
    assignments = Array(Int, n)
    distances = Array(Float64, k)

    # Initialize the cluster centers
    initialize_centers!(x, centers, n, p, k)

    # Initialize the assignment of points to clusters
    update_assignments!(x, centers, distances, assignments, n, p, k)

    # Iterate until convergence
    while !converged && iter < max_iter
        # Increment the iteration counter
        iter += 1

        # Recompute cluster assignments given current centers
        update_centers!(x, centers, assignments, n, p, k)

        # Reassign points to the closest cluster
        update_assignments!(x, centers, distances, assignments, n, p, k)

        # Update the RSS values for previous and current cluster assignments
        previous_rss = current_rss
        current_rss = rss(x, centers, assignments, n, p)

        # Assess convergence
        if previous_rss - current_rss <= tolerance
            converged = true
        end
    end

    return KMeansOutput(assignments,
                        centers,
                        iter,
                        current_rss,
                        converged)
end

function k_means{T <: Real}(x::Matrix{T}, k::Integer)
    k_means(x, k, Options())
end
