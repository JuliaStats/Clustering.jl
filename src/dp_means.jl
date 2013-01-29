# Integrate with k-means better
function dp_means{T <: Real}(data::Matrix{T}, lambda::Real, opts::Options)
    @defaults opts tolerance => 10e-8
    @defaults opts max_iter => 1_000

    # Keep track of the number of data points
    n, p = size(data)

    # Initially assign all data points to a single cluster
    k = 1
    assignments = ones(Int64, n)
    centers = mean(data, 1)
    distances = Array(Float64, k)

    # Keep working until convergence or exceeding bound on iterations
    converged, iter = false, 0

    # Keep track of error in system
    ss_old, ss_new = Inf, Inf

    while !converged && iter < max_iter
        iter += 1

        # Update cluster assignments
        for i in 1:n
            for cluster in 1:k
                d = 0.0
                for j in 1:p
                    d += (data[i, j] - centers[cluster, j])^2
                end
                distances[cluster] = d
            end
            d_min, ind_min = findmin(distances)
            if d_min > lambda
                k += 1
                assignments[i] = k
                centers = vcat(centers, data[i, :])
                push!(distances, 0.0)
            else
                assignments[i] = ind_min
            end
        end

        # Update cluster centers for non-empty clusters
        for cluster in 1:k
            n_obs = 0
            for i in 1:n
                if assignments[i] == cluster
                    n_obs += 1
                end
            end
            if n_obs > 0
                for j in 1:p
                    centers[cluster, j] = 0.0
                end
            end
            for i in 1:n
                if assignments[i] == cluster
                    for j in 1:p
                        centers[cluster, j] += data[i, j]
                    end
                end
            end
            for j in 1:p
                centers[cluster, j] /= n_obs
            end
        end

        # Update convergence criterion
        ss_old, ss_new = ss_new, 0.0
        for i in 1:n
            for j in 1:p
                ss_new += (data[i, j] - centers[assignments[i], j])^2
            end
        end
        ss_change = ss_old - ss_new
        if !isnan(ss_change) && ss_change < tolerance
            converged = true
        end
    end

    results = DPMeansOutput(assignments,
                            centers,
                            k,
                            iter,
                            ss_new,
                            converged)

    return results
end

function dp_means{T <: Real}(data::Matrix{T}, lambda::Real)
    dp_means(data, lambda, Options())
end
