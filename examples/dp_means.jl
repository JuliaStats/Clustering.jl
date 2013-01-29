using Clustering, DataFrames

n = 1_000

srand(1)

data = randn(n, 2)

centers = [0.0 0.0;
           5.0 5.0;
           10.0 0.0;
           15.0 -5.0;]

assignments = zeros(Int64, n)

for i = 1:n
    assignments[i] = rand(1:4)
    data[i, :] += centers[assignments[i], :]
end

results = dp_means(data, 50.0)

df = DataFrame()
df["Label"] = assignments
df["Cluster"] = results.assignments
by(df, ["Label", "Cluster"], nrow)

# Should really look at active clusters, not created clusters
