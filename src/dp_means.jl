# Currently assumes data is 2D.
# Tolerance good one?

function dp_means(data, lambda, max_iterations, tolerance)
  n = size(data, 1)
  k = 1
  assignments = ones(n)
  mu_x = [mean(data, 1)[1]]
  mu_y = [mean(data, 1)[2]]
  
  converged = false
  iteration = 0
  
  ss_old = Inf
  ss_new = Inf
  
  while !converged && iteration < max_iterations
    iteration = iteration + 1
    
    for i = 1:n
      distances = zeros(k)
      
      for j = 1:k
        distances[j] = (data[i, 1] - mu_x[j])^2 + (data[i, 2] - mu_y[j])^2
      end
      
      if min(distances) > lambda
        k = k + 1
        assignments[i] = k
        mu_x = [mu_x, data[i, 1]]
        mu_y = [mu_y, data[i, 2]]
      else
        assignments[i] = find(distances .== min(distances))[1]
      end
    end
    
    for j = 1:k
      if sum(assignments .== j) > 0
        indices = find(assignments .== j)
        mu_x[j] = mean(data[indices, 1])
        mu_y[j] = mean(data[indices, 2])
      end
    end
    
    ss_old = ss_new
    ss_new = 0
    
    for i = 1:n
      ss_new = ss_new + (data[i, 1] - mu_x[assignments[i]])^2 + (data[i, 2] - mu_y[assignments[i]])^2
    end
    
    ss_change = ss_old - ss_new
    
    if !isnan(ss_change) && ss_change < tolerance
      converged = true
    end
  end
  
  centers = Dict()
  centers[:x] = mu_x
  centers[:y] = mu_y
  
  results = Dict()
  results[:centers] = centers
  results[:assignments] = assignments
  results[:k] = k
  results[:iterations] = iteration
  results[:converged] = converged
  results[:ss] = ss_new
  
  results
end

dp_means(data, lambda) = dp_means(data, lambda, 1000, 10e-3)
