using BenchmarkTools
using Dates, Distances

include("benchmarks.jl")

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10.0 # Long enough

# Tuning
tunefilename = joinpath(@__DIR__, "params.json")
if !isfile(tunefilename)
    tuning = tune!(SUITE; verbose = true);
    BenchmarkTools.save(tunefilename, params(SUITE))
end
loadparams!(SUITE, BenchmarkTools.load(tunefilename)[1], :evals, :samples);

# Run and judge
results = run(SUITE; verbose = true)
# save results to JSON file
BenchmarkTools.save(joinpath(@__DIR__, "clustering_benchmark_"*Dates.format(now(), "yyyymmdd-HHMM")*".json"),
                    results)

@show results
