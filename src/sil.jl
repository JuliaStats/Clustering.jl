
function silhouettes{T<:FloatingPoint}(assignments::Vector{Int}, dist::Matrix{T})
    n = size(dist, 1)
    sils = Array(Float64, n)
    k = size(unique(assignments), 1)

    for i = 1:n
        scores = zeros(Float64, k)
        counts = zeros(Int, k)

        for j in 1:n
            m = assignments[j]
            scores[m] += dist[j,i]
            counts[m] += 1
        end
        scores ./= counts

        a = scores[assignments[i]]
        splice!(scores, assignments[i])
        b = minimum(scores)
        sils[i] = (b-a) / max(a,b)
    end
    sils
end

