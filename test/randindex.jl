# Test variational information

using Base.Test
using Clustering

a1 = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
a2 = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
a3 = [3, 3, 3, 2, 2, 2, 1, 1, 1, 1]

(ARI, RI, MI, HI) = randindex(a1, a1)

@test_approx_eq_eps ARI 1.0 1.0e-12
@test_approx_eq_eps RI  1.0 1.0e-12
@test_approx_eq_eps MI  0.0 1.0e-12
@test_approx_eq_eps HI  1.0 1.0e-12


(ARI, RI, MI, HI) = randindex(a1, a3)

@test_approx_eq_eps ARI 1.0 1.0e-12
@test_approx_eq_eps RI  1.0 1.0e-12
@test_approx_eq_eps MI  0.0 1.0e-12
@test_approx_eq_eps HI  1.0 1.0e-12


(ARI, RI, MI, HI) = randindex(a1, a2)

@test_approx_eq_eps ARI 0.403669 1.0e-5
@test_approx_eq_eps RI  0.711111 1.0e-5
@test_approx_eq_eps MI  0.288888 1.0e-5
@test_approx_eq_eps HI  0.422222 1.0e-5

@test randindex(a1, a2) == randindex(a2, a1)