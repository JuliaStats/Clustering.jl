
type MeanShiftResult{T <: FloatingPoint} <: ClusteringResult
  centers::Matrix{T}          # cluster centers (d x k)
  assignments::Vector{Int64}  # assignments (n)
  counts::Vector{Int64}       # number of samples assigned to each cluster (k)
  bandwidth::Vector{Float64}  # bandwidth (by default, 10 percent of the data range)
  scaled::Bool                # whether data is scaled or not
  scaledby::Vector{T}         # scaling factors (d)
end

#1-d profile
function g1{T <: FloatingPoint}(xi::Vector{T}, x::T, h::Float64)
  (1/2)*exp(-1/2*((x-xi)/h).^2)
end
#multi-d profile
function gd{T <: FloatingPoint}(Xi::Matrix{T},x::Vector{T},h::Vector{Float64})
  d = size(x,1)
  k = ones(T,size(Xi,1))
  for j in 1:d
    @inbounds k = k .* g1(Xi[:,j],x[j],h[j])
  end
  return k
end

# mean shift base function
function ms{T <: FloatingPoint}(X::Matrix{T}, x::Vector{T}, h::Vector{Float64})
  d = size(X,2)
  g = gd(X,x,h)

  result = zeros(T,d)
  @inbounds @simd for j in 1:d
    result[j] = sum( X[:,j].*g)/sum(g)
  end
  return result
end

function ms{T <: Real}(X::Matrix{T}, x::Vector{T}, h::Float64)
  d = size(x,1)
  ms(X,x,Float64[h for j in 1:d])
end

function msrep{T <: FloatingPoint}(X::Matrix{T}, x::Vector{T}, h::Vector{Float64}; threshold::Float64 = eps(), iter::Int64 = 200)
  s = 0
  thresholds = Float64[threshold for j in 1:iter]
  th = zeros(iter)
  m = x
  n,d = size(X)
  for j in 1:iter
    m = ms(X,x,h)
    th[j] = sqeuclidean(m, x)/sqeuclidean(x,zeros(d))
    if th[j] < threshold
      s = j
      break
    end
    x = m
  end
  m
end

function msrep{T <: FloatingPoint}(X::Matrix{T}, x::Vector{T}, h::Float64; threshold::Float64 = eps(), iter::Int64 = 200)
  d = size(X,2)
  msrep(X,x,Float64[h for j in 1:d], threshold=threshold,iter=iter)
end

#mean shift clustering
function meanshift{T <: FloatingPoint}(
  X::Matrix{T},
  h::Vector{T};
  subset::Vector{Int64}=[1:size(X,1);],
  threshold1::Float64 = 0.0001,
  threshold2::Float64 = sqrt(eps()),
  scaled::Bool = true,
  iter::Int64=200)

  n,d = size(X)

  all(1 .<= subset .<= n) ? nothing : error("subset must be values in [1:n]")

  #range of data
  s1 = ones(T,d)
  if scaled
    s1 =  mapslices( (x) -> spannorm_dist(x,zeros(size(x,1))), X ,1)[:]

    #scale data to lie by its range
    X = mapslices( (x) -> x./s1, X ,2)
  end

  finals = zeros(T,d,n)
  ncluster = 0

  savecluster = zeros(T,d,0)
  clusterlabel = zeros(Int64,n)
  counts = Array(Int64,0)

  for i in subset
    finals[:,i] = msrep(X,X[i,:][:],h,threshold=threshold2,iter=iter)
    clusterdist = zeros(ncluster)
    if ncluster >= 1
      for j in 1:ncluster
        @inbounds clusterdist[j] = sqeuclidean(savecluster[:,j],finals[:,i])/sqeuclidean(savecluster[:,j],zeros(d))
      end
    end
    if ncluster == 0 || minimum(clusterdist) > threshold1
      ncluster += 1
      push!(counts,1)
      savecluster = hcat(savecluster,finals[:,i])
      clusterlabel[i] = ncluster
    else
      clst = indmin(clusterdist)
      clusterlabel[i] = clst
      counts[clst] += 1
    end
  end
  MeanShiftResult(savecluster, clusterlabel, counts, h, scaled, s1)
end

function meanshift{T <: FloatingPoint}(
  X::Matrix{T}, 
  h::T;
  subset::Vector{Int64}=[1:size(X,1);],
  threshold1::Float64 = 0.0001,
  threshold2::Float64=sqrt(eps()),
  scaled::Bool = true,
  iter::Int64=200)

  d = size(X,2)
  meanshift(X,Float64[h for j in 1:d],subset=subset,threshold1=threshold1,threshold2=threshold2,scaled=scaled,iter=iter)
end

function meanshift{T <: FloatingPoint}(
  X::Matrix{T};
  subset::Vector{Int64}=[1:size(X,1);],
  threshold1::Float64 = 0.0001,
  threshold2::Float64=sqrt(eps()),
  scaled::Bool = true,
  iter::Int64=200)

  n,d = size(X)

  s1 =  mapslices( (x) -> spannorm_dist(x,zeros(n)), X ,1)[:]
  h = scaled ? Float64[0.1 for j in 1:d] : s1/10
  meanshift(X,h,subset=subset,threshold1=threshold1,threshold2=threshold2,scaled=scaled,iter=iter)
end

function modedetect{T <: FloatingPoint}(x::Vector{T}; maxsamples::Int64=1000)
  d = size(x,1)
  subset = [1:d;]
  if maxsamples < d
    subset = sample(1:d,maxsamples,replace=false)
  end
  meanshift(hcat(x,x),subset=subset,scaled=false).centers'[:,1]
end

