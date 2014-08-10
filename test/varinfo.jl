# Test variational information

using Base.Test
using Clustering

a1 = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
a2 = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

@test_approx_eq_eps varinfo(3, a1, 3, a1) 0.0 1.0e-12
@test_approx_eq_eps varinfo(2, a2, 2, a2) 0.0 1.0e-12

v = varinfo(3, a1, 2, a2)
v_ = varinfo(2, a2, 3, a1)
@test 0.0 < v < log(3)
@test_approx_eq v v_

a1 = [1, 2, 3, 4, 5]
a2 = [1, 1, 1, 1, 1]

@test_approx_eq varinfo(5, a1, 1, a2) log(5)
@test_approx_eq varinfo(1, a2, 5, a1) log(5)


a1 = [1, 1, 1, 2, 2, 2]
a2 = [2, 2, 2, 1, 1, 1]
@test_approx_eq_eps varinfo(2, a1, 2, a2) 0.0 1.0e-12
@test_approx_eq_eps varinfo(2, a1, 3, a2) 0.0 1.0e-12
@test_approx_eq_eps varinfo(4, a1, 3, a2) 0.0 1.0e-12
