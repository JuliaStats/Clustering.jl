function dp_means(data::Matrix{Float64}, lambda::Float64, tolerance::Float64, max_iterations::Int64)
  # Keep track of the number of data points
  n = size(data, 1)

  # Initially assign all data points to a single cluster
  k = 1
  assignments = ones(Int64, n)
  centers = mean(data, 1)

  # Keep working until convergence or exceeding bound on iterations
  converged = false
  iteration = 0

  # Keep track of error in system
  ss_old = Inf
  ss_new = Inf

  while !converged && iteration < max_iterations
    iteration += 1

    # Update cluster assignments
    for i = 1:n
      distances = zeros(k)

      for cluster_index = 1:k
        distances[cluster_index] = norm(data[i, 1] - centers[cluster_index, :])#^2
      end

      if min(distances) > lambda
        k += 1
        assignments[i] = k
        centers = vcat(centers, data[i, :])
      else
        assignments[i] = findfirst(distances .== min(distances))
      end
    end

    # Update cluster centers for non-empty clusters
    for cluster_index = 1:k
      if sum(assignments .== cluster_index) > 0
        indices = find(assignments .== cluster_index)
        centers[cluster_index, :] = mean(data[indices, :], 1)
      end
    end

    # Update convergence criterion
    ss_old = ss_new
    ss_new = 0.0

    for i = 1:n
      ss_new += norm(data[i, :] - centers[assignments[i], :])#^2
    end

    ss_change = ss_old - ss_new

    if !isnan(ss_change) && ss_change < tolerance
      converged = true
    end
  end

  results = DPMeansOutput(assignments, centers, k, iteration, ss_new, converged)

  return results
end

function dp_means(data::Matrix{Float64}, lambda::Float64)
  dp_means(data, lambda, 10e-8, 1_000)
end

function dp_means(df::DataFrame, lambda::Float64)
  dp_means(matrix(df), lambda, 10e-8, 1_000)
end
