## source this file and run compare() after reloading test/hclust.jl in Julia to check
## we get the same answers as R does.  Clustering is tricky...

compare <- function(method="complete") {
  d <- scan("hclust.txt")
  n <- sqrt(length(d))
  d <- matrix(d, n, n)
  merge <- t(matrix(scan(paste("merge-", method, ".txt", sep="")), 2, n-1))
  height <- scan(paste("height-", method, ".txt", sep=""))
  h <- hclust(as.dist(d), method)
  cat("Heights: ", sum(abs(h$height - height)), "\n")
  diffids <- merge[,1] != h$merge[,1]
  print(cbind(1:(n-1), merge, h$merge, h$height)[diffids,])
}

for (m in c("single", "complete", "average"))
    compare(m)
