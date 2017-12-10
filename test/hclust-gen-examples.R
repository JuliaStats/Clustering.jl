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

catExample <- function(h, D, method) {
  cat("Dict{String,Any}(\n\"method\" => :")
  cat(method)
  cat(",\n")
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
  cat("\n)")
}

catMethodExamples <- function(method="single") {
  for (i in 2:20) {
    for (j in 1:3) { # three examples of each size
      D = matrix(rnorm(i*i), i) * matrix(sample(c(1,0), i*i, replace=TRUE), i) + matrix(rnorm(i*i), i)*0.01
      D = D + t(D)
      catExample(hclust(as.dist(D), method), D, method)
      cat(",\n")
      catExample(hclust(as.dist(abs(D)), method), abs(D), method)
      cat(",\n")
    }
  }
}

# save a Julia file full of test examples
set.seed(1)
sink("hclust-generated-examples.jl")
cat("examples = [")
catMethodExamples("complete")
catMethodExamples("average")
catMethodExamples("single")
cat("]\n")
sink()
