# Compute RSS, which is the squared Euclidean distance between every point
# and its assigned center
function rss(x::Matrix{Float64},
             assignments::Vector{Int64},
             centers::Matrix{Float64})

  residuals = 0.0

  for i = 1:size(x, 1)
    residuals += norm(x[i, :] - centers[assignments[i], :])^2
  end

  return residuals
end

function k_means(x::Matrix{Float64}, k::Int64, tolerance::Float64, max_iter::Int64)
  # Keep track of the number of data points and their dimensionality
  n = size(x, 1)
  p = size(x, 2)

  # Random initializations of assignments
  assignments = zeros(Int64, n)
  for i in 1:n
    assignments[i] = randi(k)
  end

  # Compute centers of initial clusters.
  centers = zeros(k, p)

  # Run until convergence or until the number of iterations is too high
  converged = false
  iter = 0

  # Stop when current_rss and previous_rss are within tolerance of each other
  previous_rss = Inf
  current_rss = Inf
  delta_rss = Inf

  while !converged && iter < max_iter
    # Increment the iteration counter
    iter += 1

    # Recompute cluster assignments given current centers
    for cluster_index = 1:k
      indices = find(assignments .== cluster_index)
      centers[cluster_index, :] = mean(x[indices, :], 1)
    end

    # Reassign points to the closest cluster
    for i = 1:n
      distances = zeros(k)
      for cluster_index in 1:k
        distances[cluster_index] = norm(x[i, :] - centers[cluster_index, :])
      end
      assignments[i] = findfirst(distances .== min(distances))
    end

    # Update the RSS values for previous and current cluster assignments
    previous_rss = current_rss
    current_rss = rss(x, assignments, centers)
    delta_rss = previous_rss - current_rss

    if delta_rss <= tolerance
      converged = true
    end
  end

  results = KMeansOutput(assignments,
                         centers,
                         iter,
                         current_rss,
                         converged)

  return results
end

function k_means(x::Matrix{Float64}, k::Int64)
  k_means(x, k, 10e-8, 1_000)
end

function k_means(df::DataFrame, k::Int64)
  k_means(matrix(df), k, 10e-8, 1_000)
end
