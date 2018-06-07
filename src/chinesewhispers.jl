

# Abstractions proposed in https://github.com/JuliaLang/julia/issues/26613
colinds(A::AbstractMatrix)  = indices(A,2)

rowinds(A::AbstractMatrix, col::Integer) = indices(A,1)
rowinds(A::SparseMatrixCSC, col::Integer) =	rowvals(A)[nzrange(A, col)]

type ChineseWhispersResult <: ClusteringResult
    assignments::Vector{Int}   # assignments (n)
    counts::Vector{Int}        # number of samples assigned to each cluster (k)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end

function ChineseWhispersResult(raw_assignments::Associative, iterations, converged)
	raw_labels = getindex.(raw_assignments, 1:length(raw_assignments))
	normalised_names = Dict{eltype(raw_labels), Int}()
    counts = Int[]
    assignments = Vector{Int}(length(raw_labels))
    for (node, raw_lbl) in enumerate(raw_labels)
        name = get!(normalised_names, raw_lbl) do
            push!(counts, 0)
            length(counts) #Normalised name is next usused integer
        end
        
		counts[name]+=1
        assignments[node]=name
    end
    ChineseWhispersResult(assignments, counts, iterations, converged)
end


function chinese_whispers(sim::AbstractMatrix, max_iter=100; verbose=false)
    node_labels = DefaultDict{Int,Int}(identity; passkey=true)
    # Initially all nodes are labelled with their own ID. (nclusters==nnodes)

    for ii in 1:max_iter
        changed = false
        for node in shuffle(colinds(sim))
            old_lbl = node_labels[node]
            node_labels[node] = update_node_label(node, sim, node_labels)
            changed |= node_labels[node]==old_lbl
        end
        
        verbose && println("Iteration: $ii, lbls: $(node_labels)")
        
        if !changed
            return ChineseWhispersResult(node_labels, ii, true)
        end
    end
    
    ChineseWhispersResult(node_labels, max_iter, false)
end

function update_node_label(node::N, adj::AbstractMatrix{W}, node_labels::Associative{N, L}) where {N<:Integer, W<:Real, L}
	label_weights = Accumulator(L, W==Bool ? Int : W)
	
    neighbours = rowinds(adj, node)
    for neighbour in neighbours
        lbl = node_labels[neighbour]
        label_weights[lbl] += adj[node, neighbour]
    end
    
	old_lbl = node_labels[node]
	label_weights[old_lbl]+=zero(W) # Make sure at least one entry in the weights
    new_lbl, weight = first(most_common(label_weights, 1))
    if weight==0 # No connection
    	return old_lbl
	else
        return new_lbl
    end
end

