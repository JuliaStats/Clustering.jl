using Clustering
using Base.Test

# load the examples array
include("hclust-generated-examples.jl")

# test to make sure many random examples match R's implementation
for example in examples
    h = hclust(example["D"], example["method"])
    for i in 1:size(h.merge)[2]
        for j in 1:size(h.merge)[1]
            @test h.merge[j,i] == example["merge"][j,i]
        end
    end
    for i in 1:length(h.height)
        @test isapprox(h.height[i], example["height"][i], atol=1e-5)
    end
    for i in 1:length(h.order)
        @test h.order[i] == example["order"][i]
    end
end

#When a completely disconnected node is presented, ensure the initialization to 0-index does not kick-in.
@test begin
    mdist = [  0.0    Inf         Inf         Inf         Inf         Inf         Inf         Inf        Inf         Inf;
               Inf      0.0         0.108335  Inf         Inf         Inf         Inf         Inf        Inf         Inf;       
               Inf      0.108335    0.0         0.108332  Inf         Inf         Inf         Inf        Inf         Inf;      
               Inf    Inf           0.108332    0.0         0.858332    0.858332  Inf         Inf        Inf         Inf;      
               Inf    Inf         Inf           0.858332    0.0         0.673       0.716667    1.93333  Inf         Inf;      
               Inf    Inf         Inf           0.858332    0.673       0.0         0.716667  Inf        Inf         Inf;      
               Inf    Inf         Inf         Inf           0.716667    0.716667    0.0       Inf          0.716667  Inf;      
               Inf    Inf         Inf         Inf           1.93333   Inf         Inf           0.0        3.9145    Inf;      
               Inf    Inf         Inf         Inf         Inf         Inf           0.716667    3.9145     0.0         0.524999;
               Inf    Inf         Inf         Inf         Inf         Inf         Inf         Inf          0.524999    0.0]

    hc = hclust(mdist, :single)
    cutree(hc, h=1)' == [1 2 2 2 2 2 2 3 2 2]
end
