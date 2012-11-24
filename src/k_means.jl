# Compute RSS 
# Squared Euclidean distance between point and its assigned center, summed over all points.
# When current_rss and previous_rss are within tolerance, stop.

function k_means(x, k)
  n = size(x, 1)
  p = size(x, 2)
  
  # Random initializations of assignments.
  y = int((k - 1) * rand(n)) + 1
  
  # Compute centers of initial clusters.
  centers = zeros(k, p)
  
  converged = false
  
  iter = 0
  max_iter = 1000
  
  previous_rss = Inf
  current_rss = Inf
  delta_rss = Inf
  
  while delta_rss > 10e-6 && iter < max_iter
    # Recompute cluster assignments given centers.
    for j = 1:k
      indices = find(y .== j)
      centers[j, :] = mean(x[indices, :], 1)
    end
    
    # Reassign points to closest cluster.
    for i = 1:n
      distances = map(j -> norm(x[i, :] - centers[j, :]), [1:k])
      y[i] = find(distances .== min(distances))[1]
    end
    
    previous_rss = current_rss
    current_rss = rss(x, y, centers)
    delta_rss = previous_rss - current_rss
    
    iter = iter + 1
  end
  
  results = KMeansOutput()
  
  results.assignments = y
  results.centers = centers
  results.iterations = iter
  results.rss = current_rss
  
  results
end

function rss(x, y, centers)
  residuals = 0.0
  
  for i = 1:size(x, 1)
    residuals += norm(x[i, :] - centers[y[i], :])^2
  end
  
  residuals
end
