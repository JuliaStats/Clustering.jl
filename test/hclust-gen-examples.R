# used to build the test examples for hclust in hclust_generated_examples.jl
# run as: Rscript hclust_gen_examples.R

catExample <- function(h, D, method) {
  cat("Dict{Any,Any}(\n\"method\" => :")
  cat(method)
  cat(",\n")
  cat("\"D\" => [")
  for (i in 1:dim(D)[1]) {
    for (j in 1:dim(D)[2]) {
      cat(" ")
      cat(D[i,j])
    }
    cat(";")
  }
  cat("],\n")
  cat("\"merge\" => [")
  for (i in 1:dim(h$merge)[1]) {
    for (j in 1:dim(h$merge)[2]) {
      cat(" ")
      cat(h$merge[i,j])
    }
    cat(";")
  }
  cat("],\n")
  cat("\"order\" => [")
  for (i in 1:length(h$order)) {
    cat(h$order[i])
    cat(", ")
  }
  cat("],\n")
  cat("\"height\" => [")
  for (i in 1:length(h$height)) {
    cat(h$height[i])
    cat(", ")
  }
  cat("]\n),\n")
}

catMethodExamples <- function(rmethod="single", jmethod="single") {
  for (i in 2:20) {
    for (j in 1:3) { # three examples of each size
      D = matrix(rnorm(i*i), i) * matrix(sample(c(1,0), i*i, replace=TRUE), i) + matrix(rnorm(i*i), i)*0.01
      D = D + t(D)
      catExample(hclust(as.dist(D), rmethod), D, jmethod)
      catExample(hclust(as.dist(abs(D)), rmethod), abs(D), jmethod)
    }
  }
}


# save a Julia file full of test examples
set.seed(1)
sink("hclust-generated-examples.jl")
cat("examples = [")
catMethodExamples("complete","complete")
catMethodExamples("average","average")
catMethodExamples("single","single")
catMethodExamples("ward.D","ward1")
catMethodExamples("ward.D2","ward2")
cat("]\n")
sink()
