var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Clustering.jl Docs",
    "title": "Clustering.jl Docs",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Clustering.jl-package-1",
    "page": "Clustering.jl Docs",
    "title": "Clustering.jl package",
    "category": "section",
    "text": "Clustering.jl is a Julia package for data clustering. It covers three aspects related to data clustering:Clustering Algorithms, e.g. K-means, K-medoids, Affinity propagation, and DBSCAN, etc.\nClustering Initialization, e.g. K-means++ seeding.\nClustering Evaluation, e.g. Silhouettes and variational information."
},

{
    "location": "algorithms.html#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "algorithms.html#clu_algo_basics-1",
    "page": "Basics",
    "title": "Basics",
    "category": "section",
    "text": "The package implements a variety of clustering algorithms:Pages = [\"kmeans.md\", \"kmedoids.md\", \"hclust.md\", \"mcl.md\",\n         \"affprop.md\", \"dbscan.md\", \"fuzzycmeans.md\"]Most of the clustering functions in the package have a similar interface, making it easy to switch between different clustering algorithms."
},

{
    "location": "algorithms.html#Inputs-1",
    "page": "Basics",
    "title": "Inputs",
    "category": "section",
    "text": "A clustering algorithm, depending on its nature, may accept an input matrix in either of the following forms:Data matrix X of size d times n, the i-th column of X (X[:, i]) is a data point (data sample) in d-dimensional space.\nDistance matrix D of size n times n, where D_ij is the distance between the i-th and j-th points, or the cost of assigning them to the same cluster."
},

{
    "location": "algorithms.html#common_options-1",
    "page": "Basics",
    "title": "Common Options",
    "category": "section",
    "text": "Many clustering algorithms are iterative procedures. The functions share the basic options for controlling the iterations:maxiter::Integer: maximum number of iterations.\ntol::Real: minimal allowed change of the objective during convergence. The algorithm is considered to be converged when the change of objective value between consecutive iterations drops below tol.\ndisplay::Symbol: the level of information to be displayed. It may take one of the following values:\n:none: nothing is shown\n:final: only shows a brief summary when the algorithm ends\n:iter: shows the progress at each iteration"
},

{
    "location": "algorithms.html#Clustering.ClusteringResult",
    "page": "Basics",
    "title": "Clustering.ClusteringResult",
    "category": "type",
    "text": "Base type for the output of clustering algorithm.\n\n\n\n\n\n"
},

{
    "location": "algorithms.html#Clustering.nclusters-Tuple{ClusteringResult}",
    "page": "Basics",
    "title": "Clustering.nclusters",
    "category": "method",
    "text": "nclusters(R::ClusteringResult)\n\nGet the number of clusters.\n\n\n\n\n\n"
},

{
    "location": "algorithms.html#StatsBase.counts-Tuple{ClusteringResult}",
    "page": "Basics",
    "title": "StatsBase.counts",
    "category": "method",
    "text": "counts(R::ClusteringResult)\n\nGet the vector of cluster sizes.\n\ncounts(R)[k] is the number of points assigned to the k-th cluster.\n\n\n\n\n\n"
},

{
    "location": "algorithms.html#Clustering.assignments-Tuple{ClusteringResult}",
    "page": "Basics",
    "title": "Clustering.assignments",
    "category": "method",
    "text": "assignments(R::ClusteringResult)\n\nGet the vector of cluster indices for each point.\n\nassignments(R)[i] is the index of the cluster to which the i-th point is assigned.\n\n\n\n\n\n"
},

{
    "location": "algorithms.html#Results-1",
    "page": "Basics",
    "title": "Results",
    "category": "section",
    "text": "A clustering function would return an object (typically, an instance of some ClusteringResult subtype) that contains both the resulting clustering (e.g. assignments of points to the clusters) and the information about the clustering algorithm (e.g. the number of iterations and whether it converged).ClusteringResultThe following generic methods are supported by any subtype of ClusteringResult:nclusters(::ClusteringResult)\ncounts(::ClusteringResult)\nassignments(::ClusteringResult)"
},

{
    "location": "init.html#",
    "page": "Initialization",
    "title": "Initialization",
    "category": "page",
    "text": ""
},

{
    "location": "init.html#clu_algo_init-1",
    "page": "Initialization",
    "title": "Initialization",
    "category": "section",
    "text": "A clustering algorithm usually requires initialization before it could be started."
},

{
    "location": "init.html#Clustering.SeedingAlgorithm",
    "page": "Initialization",
    "title": "Clustering.SeedingAlgorithm",
    "category": "type",
    "text": "Base type for all seeding algorithms.\n\nEach seeding algorithm should implement the two functions: initseeds! and initseeds_by_costs!.\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.initseeds!",
    "page": "Initialization",
    "title": "Clustering.initseeds!",
    "category": "function",
    "text": "initseeds!(iseeds::AbstractVector{Int}, alg::SeedingAlgorithm,\n           X::AbstractMatrix)\n\nInitialize iseeds with the indices of cluster seeds for the X data matrix using the alg seeding algorithm.\n\nReturns iseeds.\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.initseeds_by_costs!",
    "page": "Initialization",
    "title": "Clustering.initseeds_by_costs!",
    "category": "function",
    "text": "initseeds_by_costs!(iseeds::AbstractVector{Int}, alg::SeedingAlgorithm,\n                    costs::AbstractMatrix)\n\nInitialize iseeds with the indices of cluster seeds for the costs matrix using the alg seeding algorithm.\n\nHere, mathrmcosts_ij is the cost of assigning points i and j to the same cluster. One may, for example, use the squared Euclidean distance between the points as the cost.\n\nReturns iseeds.\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.KmppAlg",
    "page": "Initialization",
    "title": "Clustering.KmppAlg",
    "category": "type",
    "text": "Kmeans++ seeding (:kmpp).\n\nChooses the seeds sequentially. The probability of a point to be chosen is proportional to the minimum cost of assigning it to the existing seeds.\n\nD. Arthur and S. Vassilvitskii (2007). k-means++: the advantages of careful seeding. 18th Annual ACM-SIAM symposium on Discrete algorithms, 2007.\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.KmCentralityAlg",
    "page": "Initialization",
    "title": "Clustering.KmCentralityAlg",
    "category": "type",
    "text": "K-medoids initialization based on centrality (:kmcen).\n\nChoose the k points with the highest centrality as seeds.\n\nHae-Sang Park and Chi-Hyuck Jun. A simple and fast algorithm for K-medoids clustering. doi:10.1016/j.eswa.2008.01.039\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.RandSeedAlg",
    "page": "Initialization",
    "title": "Clustering.RandSeedAlg",
    "category": "type",
    "text": "Random seeding (:rand).\n\nChooses an arbitrary subset of k data points as cluster seeds.\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.initseeds",
    "page": "Initialization",
    "title": "Clustering.initseeds",
    "category": "function",
    "text": "initseeds(alg::Union{SeedingAlgorithm, Symbol},\n          X::AbstractMatrix, k::Integer)\n\nSelect k seeds from a dn data matrix X using the alg algorithm.\n\nalg could be either an instance of SeedingAlgorithm or a symbolic name of the algorithm.\n\nReturns an integer vector of length k that contains the indices of chosen seeds.\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.initseeds_by_costs",
    "page": "Initialization",
    "title": "Clustering.initseeds_by_costs",
    "category": "function",
    "text": "initseeds_by_costs(alg::Union{SeedingAlgorithm, Symbol},\n                   costs::AbstractMatrix, k::Integer)\n\nSelect k seeds from the nn costs matrix using algorithm alg.\n\nHere, mathrmcosts_ij is the cost of assigning points i and j to the same cluster. One may, for example, use the squared Euclidean distance between the points as the cost.\n\nReturns an integer vector of length k that contains the indices of chosen seeds.\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.kmpp",
    "page": "Initialization",
    "title": "Clustering.kmpp",
    "category": "function",
    "text": "kmpp(X, k)\n\nUse Kmeans++ to choose k seeds from the dn data matrix X.\n\n\n\n\n\n"
},

{
    "location": "init.html#Clustering.kmpp_by_costs",
    "page": "Initialization",
    "title": "Clustering.kmpp_by_costs",
    "category": "function",
    "text": "kmpp_by_costs(C, k)\n\nUse Kmeans++ to choose k seeds based on the nn cost matrix C.\n\n\n\n\n\n"
},

{
    "location": "init.html#Seeding-1",
    "page": "Initialization",
    "title": "Seeding",
    "category": "section",
    "text": "Seeding is a type of clustering initialization, which provides a few seeds – points from a data set that would serve as the initial cluster centers (one for each cluster).Each seeding algorithm implemented by Clustering.jl is a subtype of SeedingAlgorithm:SeedingAlgorithm\ninitseeds!\ninitseeds_by_costs!There are several seeding methods described in the literature. Clustering.jl implements three popular ones:KmppAlg\nKmCentralityAlg\nRandSeedAlgFor convenience, the package also defines the two wrapper methods that take the name of the seeding algorithm and the number of clusters and take care of allocating iseeds and applying the proper SeedingAlgorithm instance:initseeds\ninitseeds_by_costsIn practice, we found that Kmeans++ is the most effective seeding method. To simplify its usage we provide:kmpp\nkmpp_by_costs"
},

{
    "location": "kmeans.html#",
    "page": "K-means",
    "title": "K-means",
    "category": "page",
    "text": ""
},

{
    "location": "kmeans.html#Clustering.kmeans",
    "page": "K-means",
    "title": "Clustering.kmeans",
    "category": "function",
    "text": "kmeans(X, k, [...])\n\nK-means clustering of the dn data matrix X (each column of X is a d-dimensional data point) into k clusters.\n\nReturns KmeansResult object.\n\nAlgorithm Options\n\ninit (defaults to :kmpp): how cluster seeds should be initialized, could be one of the following:\na Symbol, the name of a seeding algorithm (see Seeding for a list of supported methods).\nan integer vector of length k that provides the indices of points to use as initial seeds.\nweights: n-element vector of point weights (the cluster centers are the weighted means of cluster members)\nmaxiter, tol, display: see common options\n\n\n\n\n\n"
},

{
    "location": "kmeans.html#Clustering.KmeansResult",
    "page": "K-means",
    "title": "Clustering.KmeansResult",
    "category": "type",
    "text": "The output of K-means algorithm.\n\nSee also: kmeans, kmeans!.\n\n\n\n\n\n"
},

{
    "location": "kmeans.html#Clustering.kmeans!",
    "page": "K-means",
    "title": "Clustering.kmeans!",
    "category": "function",
    "text": "kmeans!(X, centers; [kwargs...])\n\nUpdate the current cluster centers (dk matrix, where d is the dimension and k the number of centroids) using the dn data matrix X (each column of X is a d-dimensional data point).\n\nReturns KmeansResult object.\n\nSee kmeans for the description of optional kwargs.\n\n\n\n\n\n"
},

{
    "location": "kmeans.html#K-means-1",
    "page": "K-means",
    "title": "K-means",
    "category": "section",
    "text": "K-means is a classical method for clustering or vector quantization. It produces a fixed number of clusters, each associated with a center (also known as a prototype), and each data point is assigned to a cluster with the nearest center.From a mathematical standpoint, K-means is a coordinate descent algorithm that solves the following optimization problem:textminimize  sum_i=1^n  mathbfx_i - boldsymbolmu_z_i ^2  textwrt  (boldsymbolmu z)Here, boldsymbolmu_k is the center of the k-th cluster, and z_i is an index of the cluster for i-th point mathbfx_i.kmeans\nKmeansResultIf you already have a set of initial center vectors, kmeans! could be used:kmeans!"
},

{
    "location": "kmeans.html#Examples-1",
    "page": "K-means",
    "title": "Examples",
    "category": "section",
    "text": "using Clustering\n\n# make a random dataset with 1000 random 5-dimensional points\nX = rand(5, 1000)\n\n# cluster X into 20 clusters using K-means\nR = kmeans(X, 20; maxiter=200, display=:iter)\n\n@assert nclusters(R) == 20 # verify the number of clusters\n\na = assignments(R) # get the assignments of points to clusters\nc = counts(R) # get the cluster sizes\nM = R.centers # get the cluster centersusing RDatasets, Clustering, Plots\niris = dataset(\"datasets\", \"iris\"); # load the data\n\nfeatures = collect(Matrix(iris[:, 1:4])\'); # features to use for clustering\nresult = kmeans(features, 3); # run K-means for the 3 clusters\n\n# plot with the point color mapped to the assigned cluster index\nscatter(iris.PetalLength, iris.PetalWidth, marker_z=result.assignments,\n        color=:lightrainbow, legend=false)"
},

{
    "location": "kmedoids.html#",
    "page": "K-medoids",
    "title": "K-medoids",
    "category": "page",
    "text": ""
},

{
    "location": "kmedoids.html#Clustering.kmedoids",
    "page": "K-medoids",
    "title": "Clustering.kmedoids",
    "category": "function",
    "text": "kmedoids(costs::DenseMatrix, k::Integer; ...)\n\nPerforms K-medoids clustering of n points into k clusters, given the costs matrix (nn, mathrmcosts_ij is the cost of assigning j-th point to the mediod represented by the i-th point).\n\nReturns an object of type KmedoidsResult.\n\nNote\n\nThis package implements a K-means style algorithm instead of PAM, which is considered much more efficient and reliable.\n\nAlgorithm Options\n\ninit (defaults to :kmpp): how medoids should be initialized, could be one of the following:\na Symbol indicating the name of a seeding algorithm (see Seeding for a list of supported methods).\nan integer vector of length k that provides the indices of points to use as initial medoids.\nmaxiter, tol, display: see common options\n\n\n\n\n\n"
},

{
    "location": "kmedoids.html#Clustering.kmedoids!",
    "page": "K-medoids",
    "title": "Clustering.kmedoids!",
    "category": "function",
    "text": "kmedoids!(costs::DenseMatrix, medoids::Vector{Int}; [kwargs...])\n\nPerforms K-medoids clustering starting with the provided indices of initial medoids.\n\nReturns KmedoidsResult object and updates the medoids indices in-place.\n\nSee kmedoids for the description of optional kwargs.\n\n\n\n\n\n"
},

{
    "location": "kmedoids.html#Clustering.KmedoidsResult",
    "page": "K-medoids",
    "title": "Clustering.KmedoidsResult",
    "category": "type",
    "text": "The output of kmedoids function.\n\nFields\n\nmedoids::Vector{Int}: the indices of k medoids\nassignments::Vector{Int}: the indices of clusters the points are assigned to, so that medoids[assignments[i]] is the index of the medoid for the i-th point\nacosts::Vector{T}: assignment costs, i.e. acosts[i] is the cost of assigning i-th point to its medoid\ncounts::Vector{Int}: cluster sizes\ntotalcost::Float64: total assignment cost (the sum of acosts)\niterations::Int: the number of executed algorithm iterations\nconverged::Bool: whether the procedure converged\n\n\n\n\n\n"
},

{
    "location": "kmedoids.html#K-medoids-1",
    "page": "K-medoids",
    "title": "K-medoids",
    "category": "section",
    "text": "K-medoids is a clustering algorithm that works by finding k data points (called medoids) such that the total cost or total distance between each data point and the closest medoid is minimal.kmedoids\nkmedoids!\nKmedoidsResult"
},

{
    "location": "hclust.html#",
    "page": "Hierarchical Clustering",
    "title": "Hierarchical Clustering",
    "category": "page",
    "text": ""
},

{
    "location": "hclust.html#Clustering.hclust",
    "page": "Hierarchical Clustering",
    "title": "Clustering.hclust",
    "category": "function",
    "text": "hclust(d::AbstractMatrix; [linkage], [uplo])\n\nPerform hierarchical clustering using the distance matrix d and the cluster linkage function.\n\nReturns the dendrogram as an object of type Hclust.\n\nArguments\n\nd::AbstractMatrix: the pairwise distance matrix. d_ij is the distance  between i-th and j-th points.\nlinkage::Symbol: cluster linkage function to use. linkage defines how the distances between the data points are aggregated into the distances between the clusters. Naturally, it affects what clusters are merged on each iteration. The valid choices are:\n:single (the default): use the minimum distance between any of the cluster members\n:average: use the mean distance between any of the cluster members\n:complete: use the maximum distance between any of the members\n:ward: the distance is the increase of the average squared distance of a point to its cluster centroid after merging the two clusters\n:ward_presquared: same as :ward, but assumes that the distances in d are already squared.\nuplo::Symbol (optional): specifies whether the upper (:U) or the lower (:L) triangle of d should be used to get the distances. If not specified, the method expects d to be symmetric.\n\n\n\n\n\n"
},

{
    "location": "hclust.html#Clustering.Hclust",
    "page": "Hierarchical Clustering",
    "title": "Clustering.Hclust",
    "category": "type",
    "text": "The output of hclust, hierarchical clustering of data points.\n\nProvides the bottom-up definition of the dendrogram as the sequence of merges of the two lower subtrees into a higher level subtree.\n\nThis type mostly follows R\'s hclust class.\n\nFields\n\nmerges::Matrix{Int}: N2 matrix encoding subtree merges:\neach row specifies the left and right subtrees (referenced by their ids) that are merged\nnegative subtree id denotes the leaf node and corresponds to the data point at position -id\npositive id denotes nontrivial subtree (the row merges[id, :] specifies its left and right subtrees)\nlinkage::Symbol: the name of cluster linkage function used to construct the hierarchy (see hclust)\nheights::Vector{T}: subtree heights, i.e. the distances between the left  and right branches of each subtree calculated using the specified linkage\norder::Vector{Int}: the data point indices ordered so that there are no  intersecting branches on the dendrogram plot. This ordering also puts  the points of the same cluster close together.\n\nSee also: hclust.\n\n\n\n\n\n"
},

{
    "location": "hclust.html#Clustering.cutree",
    "page": "Hierarchical Clustering",
    "title": "Clustering.cutree",
    "category": "function",
    "text": "cutree(hclu::Hclust; [k], [h])\n\nCuts the hclu dendrogram to produce clusters at the specified level of granularity.\n\nReturns the cluster assignments vector z (z_i is the index of the cluster for the i-th data point).\n\nArguments\n\nk::Integer (optional) the number of desired clusters.\nh::Real (optional) the height at which the tree is cut.\n\nIf both k and h are specified, it\'s guaranteed that the number of clusters is not less than k and their height is not above h.\n\nSee also: hclust\n\n\n\n\n\n"
},

{
    "location": "hclust.html#Hierarchical-Clustering-1",
    "page": "Hierarchical Clustering",
    "title": "Hierarchical Clustering",
    "category": "section",
    "text": "Hierarchical clustering algorithms build a dendrogram of nested clusters by repeatedly merging or splitting clusters.The hclust function implements several classical algorithms for hierarchical clustering (the algorithm to use is defined by the linkage parameter):hclust\nHclustusing Clustering\nD = rand(1000, 1000);\nD += D\'; # symmetric distance matrix (optional)\nresult = hclust(D, linkage=:single)The resulting dendrogram could be converted into disjoint clusters with the help of cutree function.cutree"
},

{
    "location": "mcl.html#",
    "page": "MCL (Markov Cluster Algorithm)",
    "title": "MCL (Markov Cluster Algorithm)",
    "category": "page",
    "text": ""
},

{
    "location": "mcl.html#Clustering.mcl",
    "page": "MCL (Markov Cluster Algorithm)",
    "title": "Clustering.mcl",
    "category": "function",
    "text": "mcl(adj::AbstractMatrix; [kwargs...])\n\nPerform MCL (Markov Cluster Algorithm) clustering using nn adjacency (points similarity) matrix adj.\n\nReturns MCLResult object.\n\nAlgorithm Options\n\nadd_loops::Bool (enabled by default): whether the edges of weight 1.0 from the node to itself should be appended to the graph\nexpansion::Number (defaults to 2): MCL expansion constant\ninflation::Number (defaults to 2): MCL inflation constant\nsave_final_matrix::Bool (disabled by default): whether to save the final equilibrium state in the mcl_adj field of the result; could provide useful diagnostic if the method doesn\'t converge\nprune_tol::Number: pruning threshold\ndisplay::Symbol (defaults to :none): :none for no output or :verbose for diagnostic messages\nmax_iter, tol: see common options\n\nReferences\n\nStijn van Dongen, \"Graph clustering by flow simulation\", 2001\n\nOriginal MCL implementation.\n\n\n\n\n\n"
},

{
    "location": "mcl.html#Clustering.MCLResult",
    "page": "MCL (Markov Cluster Algorithm)",
    "title": "Clustering.MCLResult",
    "category": "type",
    "text": "The output of mcl function.\n\nFields\n\nmcl_adj::AbstractMatrix: the final MCL adjacency matrix (equilibrium state matrix if the algorithm converged), empty if save_final_matrix option is disabled\nassignments::Vector{Int}: indices of the points clusters. assignments[i] is the index of the cluster for the i-th point  (0 if unassigned)\ncounts::Vector{Int}: the k-length vector of cluster sizes\nnunassigned::Int: the number of standalone points not assigned to any cluster\niterations::Int: the number of elapsed iterations\nrel_Δ::Float64: the final relative Δ\nconverged::Bool: whether the method converged\n\n\n\n\n\n"
},

{
    "location": "mcl.html#MCL-(Markov-Cluster-Algorithm)-1",
    "page": "MCL (Markov Cluster Algorithm)",
    "title": "MCL (Markov Cluster Algorithm)",
    "category": "section",
    "text": "Markov Cluster Algorithm works by simulating a stochastic (Markov) flow in a weighted graph, where each node is a data point, and the edge weights are defined by the adjacency matrix. ... When the algorithm converges, it produces the new edge weights that define the new connected components of the graph (i.e. the clusters).mcl\nMCLResult"
},

{
    "location": "affprop.html#",
    "page": "Affinity Propagation",
    "title": "Affinity Propagation",
    "category": "page",
    "text": ""
},

{
    "location": "affprop.html#Clustering.affinityprop",
    "page": "Affinity Propagation",
    "title": "Clustering.affinityprop",
    "category": "function",
    "text": "affinityprop(S::DenseMatrix; [maxiter=200], [tol=1e-6], [damp=0.5],\n             [display=:none])\n\nPerform affinity propagation clustering based on a similarity matrix S.\n\nS_ij (i  j) is the similarity (or the negated distance) between the i-th and j-th points, S_ii defines the availability of the i-th point as an exemplar.\n\nReturns an instance of AffinityPropResult.\n\nMethod parameters\n\ndamp::Real: the dampening coefficient, 0  mathrmdamp  1. Larger values indicate slower (and probably more stable) update. mathrmdamp = 0 disables dampening.\nmaxiter, tol, display: see common options\n\nNotes\n\nThe implementations is based on the following paper:\n\nBrendan J. Frey and Delbert Dueck. Clustering by Passing Messages Between Data Points. Science, vol 315, pages 972-976, 2007.\n\n\n\n\n\n"
},

{
    "location": "affprop.html#Clustering.AffinityPropResult",
    "page": "Affinity Propagation",
    "title": "Clustering.AffinityPropResult",
    "category": "type",
    "text": "The result of affinity propagation clustering (affinityprop).\n\nFields\n\nexemplars::Vector{Int}: indices of exemplars (cluster centers)\nassignments::Vector{Int}: cluster assignments for each data point\niterations::Int: number of iterations executed\nconverged::Bool: converged or not\n\n\n\n\n\n"
},

{
    "location": "affprop.html#Affinity-Propagation-1",
    "page": "Affinity Propagation",
    "title": "Affinity Propagation",
    "category": "section",
    "text": "Affinity propagation is a clustering algorithm based on message passing between data points. Similar to K-medoids, it looks at the (dis)similarities in the data, picks one exemplar data point for each cluster, and assigns every point in the data set to the cluster with the closest exemplar.affinityprop\nAffinityPropResult"
},

{
    "location": "dbscan.html#",
    "page": "DBSCAN",
    "title": "DBSCAN",
    "category": "page",
    "text": ""
},

{
    "location": "dbscan.html#DBSCAN-1",
    "page": "DBSCAN",
    "title": "DBSCAN",
    "category": "section",
    "text": "Density-based Spatial Clustering of Applications with Noise (DBSCAN) is a data clustering algorithm that finds clusters through density-based expansion of seed points. The algorithm was proposed in:Martin Ester, Hans-peter Kriegel, Jörg S, and Xiaowei Xu A density-based algorithm for discovering clusters in large spatial databases with noise. 1996."
},

{
    "location": "dbscan.html#Density-Reachability-1",
    "page": "DBSCAN",
    "title": "Density Reachability",
    "category": "section",
    "text": "DBSCAN\'s definition of a cluster is based on the concept of density reachability: a point q is said to be directly density reachable by another point p if the distance between them is below a specified threshold epsilon and p is surrounded by sufficiently many points. Then, q is considered to be density reachable by p if there exists a sequence p_1 p_2 ldots p_n such that p_1 = p and p_i+1 is directly density reachable from p_i.A cluster, which is a subset of the given set of points, satisfies two properties:All points within the cluster are mutually density-connected, meaning that for any two distinct points p and q in a cluster, there exists a point o sucht that both p and q are density reachable from o.\nIf a point is density-connected to any point of a cluster, it is also part of that cluster."
},

{
    "location": "dbscan.html#Clustering.dbscan",
    "page": "DBSCAN",
    "title": "Clustering.dbscan",
    "category": "function",
    "text": "dbscan(D::DenseMatrix, eps::Real, minpts::Int)\n\nPerform DBSCAN algorithm using the distance matrix D.\n\nReturns an instance of DbscanResult.\n\nAlgorithm Options\n\nThe following options control which points would be considered density reachable:\n\neps::Real: the radius of a point neighborhood\nminpts::Int: the minimum number of neighboring points (including itself)  to qualify a point as a density point.\n\n\n\n\n\ndbscan(points::AbstractMatrix, radius::Real;\n       leafsize = 20, min_neighbors = 1, min_cluster_size = 1)\n\nCluster points using the DBSCAN (density-based spatial clustering of applications with noise) algorithm.\n\nReturns the clustering as a vector of DbscanCluster objects.\n\nArguments\n\npoints: the dn matrix of points. points[:, j] is a d-dimensional coordinates of j-th point\nradius::Real: query radius\n\nAdditional keyword options to control the algorithm:\n\nleafsize::Int (defaults to 20): the number of points binned in each leaf node in the KDTree\nmin_neighbors::Int (defaults to 1): the minimum number of a core point neighbors\nmin_cluster_size::Int (defaults to 1): the minimum number of points in a valid cluster\n\nExample:\n\npoints = randn(3, 10000)\n# DBSCAN clustering, clusters with less than 20 points will be discarded:\nclusters = dbscan(points, 0.05, min_neighbors = 3, min_cluster_size = 20)\n\n\n\n\n\n"
},

{
    "location": "dbscan.html#Clustering.DbscanResult",
    "page": "DBSCAN",
    "title": "Clustering.DbscanResult",
    "category": "type",
    "text": "The output of dbscan function (distance matrix-based implementation).\n\nFields\n\nseeds::Vector{Int}: indices of cluster starting points\nassignments::Vector{Int}: vector of clusters indices, where each point was assigned to\ncounts::Vector{Int}: cluster sizes (number of assigned points)\n\nend ```\n\n\n\n\n\n"
},

{
    "location": "dbscan.html#Clustering.DbscanCluster",
    "page": "DBSCAN",
    "title": "Clustering.DbscanCluster",
    "category": "type",
    "text": "DBSCAN cluster returned by dbscan function (point coordinates-based implementation)\n\nFields\n\nsize::Int: number of points in a cluster (core + boundary)\ncore_indices::Vector{Int}: indices of points in the cluster core\nboundary_indices::Vector{Int}: indices of points on the cluster boundary\n\n\n\n\n\n"
},

{
    "location": "dbscan.html#Interface-1",
    "page": "DBSCAN",
    "title": "Interface",
    "category": "section",
    "text": "There are two implementations of DBSCAN algorithm in this package (both provided by dbscan function):Distance (adjacency) matrix-based. It requires O(N^2) memory to run. Boundary points cannot be shared between the clusters.\nAdjacency list-based. The input is the d times n matrix of point coordinates. The adjacency list is built on the fly. The performance is much better both in terms of running time and memory usage. Returns a vector of DbscanCluster objects that contain the indices of the core and boundary points, making it possible to share the boundary points between multiple clusters.dbscan\nDbscanResult\nDbscanCluster"
},

{
    "location": "fuzzycmeans.html#",
    "page": "Fuzzy C-means",
    "title": "Fuzzy C-means",
    "category": "page",
    "text": ""
},

{
    "location": "fuzzycmeans.html#Clustering.fuzzy_cmeans",
    "page": "Fuzzy C-means",
    "title": "Clustering.fuzzy_cmeans",
    "category": "function",
    "text": "fuzzy_cmeans(data::AbstractMatrix, C::Int, fuzziness::Real; [...])\n\nPerforms Fuzzy C-means clustering over the given data.\n\nReturns an instance of FuzzyCMeansResult.\n\nArguments\n\ndata::AbstractMatrix: dn data matrix. Each column represents one d-dimensional data point.\nC::Int: the number of fuzzy clusters, 2  C  n.\nfuzziness::Real: clusters fuzziness (see m in the mathematical formulation), mathrmfuzziness  1.\n\nOne may also control the algorithm via the following optional keyword arguments:\n\ndist_metric::Metric (defaults to Euclidean): the Metric object  that defines the distance between the data points\nmaxiter, tol, display: see common options\n\n\n\n\n\n"
},

{
    "location": "fuzzycmeans.html#Clustering.FuzzyCMeansResult",
    "page": "Fuzzy C-means",
    "title": "Clustering.FuzzyCMeansResult",
    "category": "type",
    "text": "The output of fuzzy_cmeans function.\n\nFields\n\ncenters::Matrix{T}: the dC matrix with columns being the centers of resulting fuzzy clusters\nweights::Matrix{Float64}: the nC matrix of assignment weights (mathrmweights_ij is the weight (probability) of assigning i-th point to the j-th cluster)\niterations::Int: the number of executed algorithm iterations\nconverged::Bool: whether the procedure converged\n\n\n\n\n\n"
},

{
    "location": "fuzzycmeans.html#fuzzy_cmeans_def-1",
    "page": "Fuzzy C-means",
    "title": "Fuzzy C-means",
    "category": "section",
    "text": "Fuzzy C-means is a clustering method that provides cluster membership weights instead of \"hard\" classification (e.g. K-means).From a mathematical standpoint, fuzzy C-means solves the following optimization problem:argmin_C  sum_i=1^n sum_j=1^c w_ij^m  mathbfx_i - mathbfc_j ^2 \ntextwhere w_ij = left(sum_k=1^c left(fracleftmathbfx_i - mathbfc_j rightleftmathbfx_i - mathbfc_k rightright)^frac2m-1right)^-1Here, mathbfc_j is the center of the j-th cluster, w_ij is the membership weight of the i-th point in the j-th cluster, and m  1 is a user-defined fuzziness parameter.fuzzy_cmeans\nFuzzyCMeansResult"
},

{
    "location": "fuzzycmeans.html#Examples-1",
    "page": "Fuzzy C-means",
    "title": "Examples",
    "category": "section",
    "text": "using Clustering\n\n# make a random dataset with 1000 points\n# each point is a 5-dimensional vector\nX = rand(5, 1000)\n\n# performs Fuzzy C-means over X, trying to group them into 3 clusters\n# with a fuzziness factor of 2. Set maximum number of iterations to 200\n# set display to :iter, so it shows progressive info at each iteration\nR = fuzzy_cmeans(X, 3, 2, maxiter=200, display=:iter)\n\n# get the centers (i.e. weighted mean vectors)\n# M is a 5x3 matrix\n# M[:, k] is the center of the k-th cluster\nM = R.centers\n\n# get the point memberships over all the clusters\n# memberships is a 20x3 matrix\nmemberships = R.weights"
},

{
    "location": "validate.html#",
    "page": "Basics",
    "title": "Basics",
    "category": "page",
    "text": ""
},

{
    "location": "validate.html#clu_validate-1",
    "page": "Basics",
    "title": "Basics",
    "category": "section",
    "text": "Clustering.jl package provides a number of methods to evaluate the results of a clustering algorithm and/or to validate its correctness:Pages=[\"silhouette.md\", \"varinfo.md\", \"randindex.md\", \"vmeasure.md\"]"
},

{
    "location": "silhouette.html#",
    "page": "Silhouettes",
    "title": "Silhouettes",
    "category": "page",
    "text": ""
},

{
    "location": "silhouette.html#Clustering.silhouettes",
    "page": "Silhouettes",
    "title": "Clustering.silhouettes",
    "category": "function",
    "text": "silhouettes(assignments::AbstractVector, [counts,] dists)\nsilhouettes(clustering::ClusteringResult, dists)\n\nCompute silhouette values for individual points w.r.t. given clustering.\n\nReturns the n-length vector of silhouette values for each individual point.\n\nArguments\n\nassignments::AbstractVector{Int}: the vector of point assignments (cluster indices)\ncounts::AbstractVector{Int}: the optional vector of cluster sizes (how many points assigned to each cluster; should match assignments)\nclustering::ClusteringResult: the output of some clustering method\ndists::AbstractMatrix: nn matrix of pairwise distances between the points\n\n\n\n\n\n"
},

{
    "location": "silhouette.html#Silhouettes-1",
    "page": "Silhouettes",
    "title": "Silhouettes",
    "category": "section",
    "text": "Silhouettes is a method for evaluating the quality of clustering. Particularly, it provides a quantitative way to measure how well each point lies within its cluster in comparison to the other clusters. It was introduced inPeter J. Rousseeuw (1987). Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis. Computational and Applied Mathematics. 20: 53–65.The Silhouette value for the i-th data point is:s_i = fracb_i - a_imax(a_i b_i)  textwherea_i is the average distance from the i-th point to the other points in the same cluster z_i,\nb_i  min_k ne z_i b_ik, where b_ik is the average distance from the i-th point to the points in the k-th cluster.Note that s_i le 1, and that s_i is close to 1 when the i-th point lies well within its own cluster. This property allows using mean(silhouettes(assignments, counts, X)) as a measure of clustering quality. Higher values indicate better separation of clusters w.r.t. point distances.silhouettes"
},

{
    "location": "randindex.html#",
    "page": "Rand index",
    "title": "Rand index",
    "category": "page",
    "text": ""
},

{
    "location": "randindex.html#Clustering.randindex",
    "page": "Rand index",
    "title": "Clustering.randindex",
    "category": "function",
    "text": "randindex(c1, c2)\n\nCompute the tuple of Rand-related indices between the clusterings c1 and c2.\n\nThe clusterings could be either point-to-cluster assignment vectors or instances of ClusteringResult subtype.\n\nReturns a tuple of indices:\n\nHubert & Arabie Adjusted Rand index\nRand index\nMirkin\'s index\nHubert\'s index\n\n\n\n\n\n"
},

{
    "location": "randindex.html#Rand-index-1",
    "page": "Rand index",
    "title": "Rand index",
    "category": "section",
    "text": "Rand index is a measure of the similarity between the two data clusterings. From a mathematical standpoint, Rand index is related to the accuracy, but is applicable even when class labels are not used. The measure was introduced inLawrence Hubert and Phipps Arabie (1985). Comparing partitions. Journal of Classification 2 (1): 193–218See also:Meila, Marina (2003). Comparing Clusterings by the Variation of Information. Learning Theory and Kernel Machines: 173–187.randindex"
},

{
    "location": "varinfo.html#",
    "page": "Variation of Information",
    "title": "Variation of Information",
    "category": "page",
    "text": ""
},

{
    "location": "varinfo.html#Clustering.varinfo",
    "page": "Variation of Information",
    "title": "Clustering.varinfo",
    "category": "function",
    "text": "varinfo(k1::Int, a1::AbstractVector{Int}, k2::Int, a2::AbstractVector{Int})\nvarinfo(R::ClusteringResult, k0::Int, a0::AbstractVector{Int})\nvarinfo(R1::ClusteringResult, R2::ClusteringResult)\n\nCompute the variation of information between the two clusterings.\n\nEach clustering is provided either as an instance of ClusteringResult subtype or as a pair of arguments:\n\na number of clusters (k1, k2, k0)\na vector of point to cluster assignments (a1, a2, a0).\n\n\n\n\n\n"
},

{
    "location": "varinfo.html#Variation-of-Information-1",
    "page": "Variation of Information",
    "title": "Variation of Information",
    "category": "section",
    "text": "Variation of information (also known as shared information distance) is a measure of the distance between the two clusterings. It is devised from the mutual information, but it is a true metric, i.e. it is symmetric and satisfies the triangle inequality. SeeMeila, Marina (2003). Comparing Clusterings by the Variation of Information. Learning Theory and Kernel Machines: 173–187.varinfo"
},

{
    "location": "vmeasure.html#",
    "page": "V-measure",
    "title": "V-measure",
    "category": "page",
    "text": ""
},

{
    "location": "vmeasure.html#Clustering.vmeasure",
    "page": "V-measure",
    "title": "Clustering.vmeasure",
    "category": "function",
    "text": "vmeasure(assign1, assign2; [β = 1.0])\n\nV-measure between two clustering assignments.\n\nassign1 and assign2 can be either ClusteringResult instances or assignments vectors (AbstractVector{<:Integer}).\n\nThe β parameter defines trade-off between homogeneity and completeness:\n\nif β  1, completeness is weighted more strongly,\nif β  1, homogeneity is weighted more strongly.\n\nReferences\n\nAndrew Rosenberg and Julia Hirschberg, 2007. \"V-Measure: A conditional entropy-based external cluster evaluation measure\"\n\n\n\n\n\n"
},

{
    "location": "vmeasure.html#V-measure-1",
    "page": "V-measure",
    "title": "V-measure",
    "category": "section",
    "text": "V-measure can be used to compare the clustering results with the existing class labels of data points or with the alternative clustering. It is defined as the harmonic mean of homogeneity (h) and completeness (c) of the clustering:V_beta = (1+beta)frach cdot cbeta cdot h + cBoth h and c can be expressed in terms of the mutual information and entropy measures from the information theory. Homogeneity (h) is maximized when each cluster contains elements of as few different classes as possible. Completeness (c) aims to put all elements of each class in single clusters. The beta parameter (beta  0) could used to control the weights of h and c in the final measure. If beta  1, completeness has more weight, and when beta  1 it\'s homogeneity.vmeasure"
},

]}
