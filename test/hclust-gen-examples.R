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

catExample <- function(h, D, linkage, cutk, cuth) {
  cat("Dict{String,Any}(\n\"linkage\" => :", linkage, ",\n", sep="")
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
  cutt <- cutree(h, cutk, cuth)
  catVector(cutt)
  cat("\n)")
}

catMethodExamples <- function(method="single", jl_linkage=method) {
  for (i in 2:60) { # i: number of elements
    D = matrix(rnorm(i*i), i) * matrix(sample(c(1,0), i*i, replace=TRUE), i) + matrix(rnorm(i*i), i)*0.01
    D1 = D + t(D)
    h1 = hclust(as.dist(D1), method)
    if (runif(1) < 0.5) {
      cutk1 = sample.int(i, 1)
      cuth1 = NULL
    } else {
      cutk1 = NULL
      cuth1 = sample(h1$height, 1)
    }
    catExample(h1, D1, jl_linkage, cutk1, cuth1)
    cat(",\n")
    D2 = abs(D1)
    h2 = hclust(as.dist(D2), method)
    if (runif(1) < 0.5) {
      cutk2 = sample.int(i, 1)
      cuth2 = NULL
    } else {
      cutk2 = NULL
      cuth2 = sample(h2$height, 1)
    }
    catExample(h2, D2, jl_linkage, cutk2, cuth2)
    cat(",\n")
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
