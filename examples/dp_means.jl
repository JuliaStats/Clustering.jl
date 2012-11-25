load("Clustering")
using Clustering

n = 1_000

data = randn(n, 2)

centers = [0.0 0.0;
           5.0 5.0;
           10.0 0.0;
           15.0 -5.0;]

assignments = zeros(Int64, n)

for i = 1:n
  assignments[i] = randi(4)
  data[i, :] += centers[assignments[i], :]
end

results = dp_means(data, 10.0)

# Should really look at active clusters, not created clusters
results.k
