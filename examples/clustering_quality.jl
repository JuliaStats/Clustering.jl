using Plots, Clustering

## test data with 3 clusters
X = hcat([4., 5.] .+ 0.4 * randn(2, 10),
         [9., -5.] .+ 0.4 * randn(2, 5),
         [-4., -9.] .+ 1 * randn(2, 5))

## visualisation of the exemplary data
scatter(X[1,:], X[2,:],
    label = "data points",
    xlabel = "x",
    ylabel = "y",
    legend = :right,
)

nclusters = 2:5

## hard clustering quality
clusterings = kmeans.(Ref(X), nclusters)
hard_indices = [:silhouettes, :calinski_harabasz, :xie_beni, :davies_bouldin, :dunn]

kmeans_quality = Dict(
    qidx => clustering_quality.(Ref(X), clusterings, quality_index = qidx)
    for qidx in hard_indices)

plot((
    plot(nclusters, kmeans_quality[qidx],
         marker = :circle,
         title = qidx,
         label = nothing,
    ) for qidx in hard_indices)...,
    layout = (3, 2),
    xaxis = "N clusters",
    plot_title = "\"Hard\" clustering quality indices"
)

## soft clustering quality
fuzziness = 2
fuzzy_clusterings = fuzzy_cmeans.(Ref(X), nclusters, fuzziness)
soft_indices = [:calinski_harabasz, :xie_beni]

fuzzy_cmeans_quality = Dict(
    qidx => clustering_quality.(Ref(X), fuzzy_clusterings, fuzziness = fuzziness, quality_index = qidx)
    for qidx in soft_indices)

plot((
    plot(nclusters, fuzzy_cmeans_quality[qidx],
        marker = :circle,
        title = qidx,
        label = nothing,
    ) for qidx in soft_indices)...,
    layout = (2, 1),
    xaxis = "N clusters",
    plot_title = "\"Soft\" clustering quality indices"
)
