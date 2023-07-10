using Plots, Clustering, Statistics

X = Matrix{Float64}(undef,2,20)

for k in 1:10
    X[:,k] = [4,5] .+ 0.2randn(2)
end
for k in 11:15
    X[:,k] = [9,-5] .+ 0.2randn(2)
end
for k in 15:20
    X[:,k] = [-4,-9] .+ 0.5randn(2)
end


scatter(X[1,:],X[2,:],
    label = nothing,
)

##
resf = fuzzy_cmeans(X,3,2)

res = kmeans(X,3)

q = [calinski_harabasz(X,kmeans(X,k)) for k in 2:5]
q = [xie_beni(X,kmeans(X,k)) for k in 2:5]
q = [davies_bouldin(X,kmeans(X,k)) for k in 2:5]

qf = [calinski_harabasz(X,fuzzy_cmeans(X,k,2), 2) for k in 2:5]
qf = [xie_beni(X,fuzzy_cmeans(X,k,2), 2) for k in 2:5]

plot(2:5,q)


calinski_harabasz(X,res)