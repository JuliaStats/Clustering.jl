# used to build the test examples for hclust in hclust_generated_examples.jl
# run as: Rscript hclust_gen_examples.R

catVector <- function(V) {
  cat("[")
  for (i in 1:length(V)) {
    cat(V[i], ", ", sep="")
  }
  cat("]")
}

catMatrix <- function(M) {
  cat("[")
  for (i in 1:dim(M)[1]) {
    for (j in 1:dim(M)[2]) {
      cat(" ", M[i,j], sep="")
    }
    cat(";")
  }
  cat("]")
}

catExample <- function(D, method, jl_linkage) {
  n <- ncol(D)
  triD <- D[rep(1:n, times=n) < rep(1:n, each=n)]
  if (any(table(triD)>1)) {
    message("Skipping method=", method, " n=", n, ": ambiguous tree merges")
    return()
  }
  h <- hclust(as.dist(D), method)

  if (runif(1) < 0.5) {
    cutk = sample.int(n, 1)
    cuth = NULL
  } else {
    cutk = NULL
    cuth = if (n > 2) sample(h$height, 1) else h$height[[1]]
  }
  cutt <- cutree(h, cutk, cuth)

  cat("Dict{String,Any}(\n\"linkage\" => :", jl_linkage, ", \"n\" => ", ncol(D), ",\n", sep="")
  cat("\"D\" => ")
  catMatrix(D)
  cat(",\n")
  cat("\"merge\" => ")
  catMatrix(h$merge)
  cat(",\n")
  cat("\"order\" => ")
  catVector(h$order)
  cat(",\n")
  cat("\"height\" => ")
  catVector(h$height)
  cat(",\n")
  cat("\"cut_k\" => ", ifelse(is.null(cutk), "nothing", cutk),
    ", \"cut_h\" => ", ifelse(is.null(cuth), "nothing", cuth),
    ",\n", sep="")
  cat("\"cutree\" => ")
  catVector(cutt)
  cat("\n),\n")
}

catMethodExamples <- function(method="single", jl_linkage=method) {
  for (n in 2:60) { # n: number of elements
    D = matrix(rnorm(n*n), n) *
        matrix(sample(c(1,0), n*n, replace=TRUE), n) +
        matrix(rnorm(n*n), n)*0.01
    D1 = round(D + t(D), digits=8)
    catExample(D1, method, jl_linkage)
    D2 = abs(D1)
    catExample(D2, method, jl_linkage)
  }
}

# save a Julia file full of test examples
set.seed(1)
hclu_examples_file <- gzfile("data/hclust_generated_examples.jl.gz", "w")
message("Writing hclust() examples...")
sink(hclu_examples_file)
cat("examples = [")
catMethodExamples("complete")
catMethodExamples("average")
catMethodExamples("single")
catMethodExamples("ward.D", "ward_presquared")
catMethodExamples("ward.D2", "ward")
cat("]\n")
sink()
close(hclu_examples_file)
message("Done")
