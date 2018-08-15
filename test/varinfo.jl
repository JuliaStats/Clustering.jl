# Test variational information

using Test
using Clustering

@testset "varinfo() (variational information)" begin

a1 = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
a2 = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

@test varinfo(3, a1, 3, a1) ≈ 0.0 atol=1.0e-12
@test varinfo(2, a2, 2, a2) ≈ 0.0 atol=1.0e-12

v = varinfo(3, a1, 2, a2)
v_ = varinfo(2, a2, 3, a1)
@test 0.0 < v < log(3)
@test v ≈ v_

a1 = [1, 2, 3, 4, 5]
a2 = [1, 1, 1, 1, 1]

@test varinfo(5, a1, 1, a2) ≈ log(5)
@test varinfo(1, a2, 5, a1) ≈ log(5)

a1 = [1, 1, 1, 2, 2, 2]
a2 = [2, 2, 2, 1, 1, 1]
@test varinfo(2, a1, 2, a2) ≈ 0.0 atol=1.0e-12
@test varinfo(2, a1, 3, a2) ≈ 0.0 atol=1.0e-12
@test varinfo(4, a1, 3, a2) ≈ 0.0 atol=1.0e-12

end
