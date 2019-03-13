using Distributions
import Clustering: fitness

function fruspini()
    ruspini = [
        4	53;
        5	63;
        10	59;
        9	77;
        13	49;
        13	69;
        12	88;
        15	75;
        18	61;
        19	65;
        22	74;
        27	72;
        28	76;
        24	58;
        27	55;
        28	60;
        30	52;
        31	60;
        32	61;
        36	72;
        28	147;
        32	149;
        35	153;
        33	154;
        38	151;
        41	150;
        38	145;
        38	143;
        32	143;
        34	141;
        44	156;
        44	149;
        44	143;
        46	142;
        47	149;
        49	152;
        50	142;
        53	144;
        52	152;
        55	155;
        54	124;
        60	136;
        63	139;
        86	132;
        85	115;
        85	96;
        78	94;
        74	96;
        97	122;
        98	116;
        98	124;
        99	119;
        99	128;
        101	115;
        108	111;
        110	111;
        108	116;
        111	126;
        115	117;
        117	115;
        70	4;
        77	12;
        83	21;
        61	15;
        69	15;
        78	16;
        66	18;
        58	13;
        64	20;
        69	21;
        66	23;
        61	25;
        76	27;
        72	31;
        64	30
    ]
    return [ruspini[i, :] for i = 1:lastindex(ruspini, 1)]
end

function rand200()
    nd1 = MvNormal([0, 10], [1.7, 1.7])
    nd2 = MvNormal([20, 12], [0.7, 0.7])
    nd3 = MvNormal([10, 20], [1.0, 1.0])
    s = hcat(rand(nd1, 120), rand(nd2, 60), rand(nd3, 20))
    t = s'
    tobj = [t[i, :] for i = 1:lastindex(t, 1)]
    return tobj
end

@testset "Clustering Genetic Algorithms" begin
    d, r = cga(fruspini())
    @test counts(r) == [20, 23, 17, 15]
    @test 1.73 < fitness(assignments(r), d) < 1.74
    d, r = cga(rand200())
    @test counts(r) == [120, 60, 20]
    @test 1.75 < fitness(assignments(r), d) < 1.85
end
