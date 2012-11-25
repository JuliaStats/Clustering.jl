type KMeansOutput
  assignments::Vector{Int64}
  centers::Matrix{Float64}
  iterations::Int64
  rss::Float64
  converged::Bool
end

KMeansOutput() = KMeansOutput([0], [0.0], 0, 0.0, false)

function show(results::KMeansOutput)
  println(results.assignments)
  println(results.centers)
  println(results.iterations)
  println(results.rss)
  println(results.converged)
end

type DPMeansOutput
  assignments::Vector{Int64}
  centers::Matrix{Float64}
  k::Int64
  iterations::Int64
  rss::Float64
  converged::Bool
end

DPMeansOutput() = DPMeansOutput([0], [0.0], 0, 0, 0.0, false)

function show(results::DPMeansOutput)
  println(results.assignments)
  println(results.centers)
  println(results.k)
  println(results.iterations)
  println(results.rss)
  println(results.converged)
end
