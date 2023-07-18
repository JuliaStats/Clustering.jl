using Plots, Clustering

## visualisation of the exemplary data
## there are 3 real clusters

X = hcat([4., 5.] .+ 0.4 * randn(2, 10),
         [9., -5.] .+ 0.4 * randn(2, 5),
         [-4., -9.] .+ 1 * randn(2, 5))


scatter(X[1,:],X[2,:],
    label = "exemplary data points",
    xlabel = "x",
    ylabel = "y",
    legend = :right,
)

## hard clustering quality for number of clusters in 2:5

clusterings = kmeans.(Ref(X), 2:5)
hard_indices = [:silhouette, :calinski_harabasz, :xie_beni, :davies_bouldin, :dunn]

kmeans_quality = 
    Dict(qidx => clustering_quality.(Ref(X), clusterings, quality_index = qidx)
        for qidx in hard_indices
    )


p = [
    plot(2:5, [kmeans_quality[qidx] ],
        marker = :circle,
        title = string.(qidx),
        label = nothing,
    )
        for qidx in hard_indices
]
plot(p...,
    layout = (3,2),
    plot_title = "Quality indices for various number of clusters"
)

## soft clustering quality for number of clusters in 2:5

fuzziness = 2
soft_indices = [:calinski_harabasz, :xie_beni]
fuzzy_clusterings = fuzzy_cmeans.(Ref(X), 2:5, fuzziness)

fuzzy_cmeans_quality = 
    Dict(qidx => clustering_quality.(Ref(X), fuzzy_clusterings, fuzziness, quality_index = qidx)
        for qidx in soft_indices
    )


p = [
    plot(2:5, fuzzy_cmeans_quality[qidx],
        marker = :circle,
        title = string.(qidx),
        label = nothing,
    )
        for qidx in soft_indices
]
plot(p...,
    layout = (2,1),
    plot_title = "Quality indices for various number of clusters"
)



