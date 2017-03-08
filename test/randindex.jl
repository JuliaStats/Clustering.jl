# Test variational information

using Base.Test
using Clustering

a1 = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
a2 = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
a3 = [3, 3, 3, 2, 2, 2, 1, 1, 1, 1]

(ARI, RI, MI, HI) = randindex(a1, a1)

@test isapprox(ARI, 1.0, atol=1.0e-12)
@test isapprox(RI , 1.0, atol=1.0e-12)
@test isapprox(MI , 0.0, atol=1.0e-12)
@test isapprox(HI , 1.0, atol=1.0e-12)


(ARI, RI, MI, HI) = randindex(a1, a3)

@test isapprox(ARI, 1.0, atol=1.0e-12)
@test isapprox(RI , 1.0, atol=1.0e-12)
@test isapprox(MI , 0.0, atol=1.0e-12)
@test isapprox(HI , 1.0, atol=1.0e-12)


(ARI, RI, MI, HI) = randindex(a1, a2)

@test isapprox(ARI, 0.403669, atol=1.0e-5)
@test isapprox(RI , 0.711111, atol=1.0e-5)
@test isapprox(MI , 0.288888, atol=1.0e-5)
@test isapprox(HI , 0.422222, atol=1.0e-5)

@test randindex(a1, a2) == randindex(a2, a1)
