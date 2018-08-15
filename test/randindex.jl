# Test Rand index

using Test
using Clustering

@testset "randindex() (Rand index)" begin

a1 = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
a2 = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
a3 = [3, 3, 3, 2, 2, 2, 1, 1, 1, 1]

(ARI, RI, MI, HI) = randindex(a1, a1)

@test ARI ≈ 1.0 atol=1.0e-12
@test RI  ≈ 1.0 atol=1.0e-12
@test MI  ≈ 0.0 atol=1.0e-12
@test HI  ≈ 1.0 atol=1.0e-12


(ARI, RI, MI, HI) = randindex(a1, a3)

@test ARI ≈ 1.0 atol=1.0e-12
@test RI  ≈ 1.0 atol=1.0e-12
@test MI  ≈ 0.0 atol=1.0e-12
@test HI  ≈ 1.0 atol=1.0e-12


(ARI, RI, MI, HI) = randindex(a1, a2)

@test ARI ≈ 0.403669 atol=1.0e-5
@test RI  ≈ 0.711111 atol=1.0e-5
@test MI  ≈ 0.288888 atol=1.0e-5
@test HI  ≈ 0.422222 atol=1.0e-5

@test randindex(a1, a2) == randindex(a2, a1)

end
