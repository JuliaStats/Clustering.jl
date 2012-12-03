type KMeansOutput
  assignments::Vector{Int64}
  centers::Matrix{Float64}
  iterations::Int64
  rss::Float64
  converged::Bool
end

KMeansOutput() = KMeansOutput([0], [0.0], 0, 0.0, false)

function show(results::KMeansOutput)
  println("Cluster Assignments:")
  println(results.assignments)
  println()
  println("Cluster Centers:")
  println(results.centers)
  println()
  print("Number of Iterations: ")
  println(results.iterations)
  println()
  print("Final RSS: ")
  println(results.rss)
  println()
  print("Algorithm Converged: ")
  println(results.converged)
  println()
end

function repl_show(io::IO, results::KMeansOutput)
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

function repl_show(io::IO, results::DPMeansOutput)
  println(io, results.assignments)
  println(io, results.centers)
  println(io, results.k)
  println(io, results.iterations)
  println(io, results.rss)
  println(io, results.converged)
end
