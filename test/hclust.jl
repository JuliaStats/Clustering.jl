function test_hclust(d::Matrix, method=:complete)
    h = hclust(d, method)
    writedlm("merge-$method.txt", h.merge)
    writedlm("height-$method.txt", h.height)
    c = cutree(h, h=0)
    writedlm("cutree-$method.txt", c)
    d
end

function gendist(N::Int)
    d = randn(N,N)
    d += d'
    d
end

d = gendist(1000)
writedlm("hclust.txt", d)

for m in [:single, :complete, :average]
    test_hclust(d, m)
end

## now run the following command, and inspect the output
## Height should be 0 and matrix should have 0 rows

# run(`Rscript test/hclust.R`)
