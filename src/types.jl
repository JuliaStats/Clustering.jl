type KMeansOutput
  assignments
  centers
  iterations
  rss
end

KMeansOutput() = KMeansOutput(0, 0, 0, 0)
