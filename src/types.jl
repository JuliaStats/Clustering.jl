mutable struct KMeansOutput
  assignments::Vector{Int64}
  centers::Matrix{Float64}
  iterations::Int64
  rss::Float64
  converged::Bool
end

KMeansOutput() = KMeansOutput(Int64[], Float64[], 0, 0.0, false)

function show(io::IO, results::KMeansOutput)
  println(io, "Cluster Assignments:")
  println(io, results.assignments)
  println(io)
  println(io, "Cluster Centers:")
  println(io, results.centers)
  println(io)
  print(io, "Number of Iterations: ")
  println(io, results.iterations)
  println(io)
  print(io, "Final RSS: ")
  println(io, results.rss)
  println(io)
  print(io, "Algorithm Converged: ")
  println(io, results.converged)
  println(io)
end

mutable struct DPMeansOutput
  assignments::Vector{Int64}
  centers::Matrix{Float64}
  k::Int64
  iterations::Int64
  rss::Float64
  converged::Bool
end

DPMeansOutput() = DPMeansOutput(Int64[], Float64[], 0, 0, 0.0, false)

function show(io::IO, results::DPMeansOutput)
  println(io, "Cluster Assignments:")
  println(io, results.assignments)
  println(io)
  println(io, "Cluster Centers:")
  println(io, results.centers)
  println(io)
  println(io, "Cluster Created:")
  println(io, results.k)
  println(io)
  print(io, "Number of Iterations: ")
  println(io, results.iterations)
  println(io)
  print(io, "Final RSS: ")
  println(io, results.rss)
  println(io)
  print(io, "Algorithm Converged: ")
  println(io, results.converged)
end
