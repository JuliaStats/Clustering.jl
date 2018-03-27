using Base.Test
using Distances
using Clustering


@testset "basic seperated graph" begin
    eg1 = [
        0 1 0 1;
        1 0 0 0;
        0 0 0 0;
        1 0 0 0;
    ]
    @testset "$(first(vv))" for vv in [("dense", eg1), ("sparse", sparse(eg1))]
        eg = last(vv)
        res = chinese_whispers(eg)
        lbls = assignments(res)
        @test lbls[3] != lbls[1]
        @test lbls[3] != lbls[2]
        @test lbls[3] != lbls[4]
        
        @test nclusters(res) >= 2
        @test sum(counts(res)) == 4
    end
end

@testset "planar based" begin
    srand(1) # make determanistic
    coordersA = randn(10, 2)
    coordersB = randn(10, 2) .+ [5 5]

    coords = [coordersA; coordersB]';

    adj = 1./pairwise(Euclidean(), coords)
    adj[isinf.(adj)]=0 # no selfsim
    adj[rand(size(adj)).<0.6]=0 #remove some connections

    res = chinese_whispers(adj)
    lbls = assignments(res)
    @test all(lbls[1].==(lbls[1:10]))
    @test all(lbls[20].==(lbls[11:20]))
    
    @test nclusters(res) == 2
    @test counts(res) == [10, 10]
end


@testset "acts the same for all types" begin
    examples = [
        sprand(500,500,0.3),
        sprand(1500,1500,0.1).>0.5, #Boolean elements
        rand(200, 200)
    ]
	function test_assignments(x)
		srand(1)
		assignments(chinese_whispers(x))
	end

    for eg in (examples)
        eg = collect(Symmetric(eg))
        dense_res =  test_assignments(eg)
        sparse_res = test_assignments(sparse(eg))
        symetric_res = test_assignments(Symmetric(eg))
        
        @test  dense_res == sparse_res == symetric_res
    end
end
