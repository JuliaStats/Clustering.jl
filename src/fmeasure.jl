"""
    _pair_confusion_matrix(a,b) -> NTuple{4, Int64}

Compute the similarities between two clusterings by considering all pairs of samples.

Returns (TruePositive,FalsePositive,FalseNegative,TrueNegative)

"""
function _pair_confusion_matrix(a,b)
    c = counts(a, b)
    n = length(a)
    n_k = sum(c,dims=1)[:]
    n_c = sum(c,dims=2)[:]
    n_sum = sum(c.*c)
    tp = n_sum-n
    fp = sum(c*n_k)-n_sum
    fn = sum(c'*n_c)-n_sum
    tn = n^2-fp-fn-n_sum 
    return tp,fp,fn,tn
end



"""
    pair_precision(a, b) -> Float64

Compute the pair counting precision between two clustering of the same data points.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

Returns the value of the pair counting precision.

# References
> Pfitzner, Darius, Richard Leibbrandt, and David Powers. (2009).
> *Characterization and evaluation of similarity measures for pairs of clusterings.*
> Knowledge and Information Systems: 361-394.
"""
function pair_precision(a, b)
    tp,fp,fn,tn = _pair_confusion_matrix(a,b)
    return tp/(tp+fp)
end

"""
    pair_recall(a, b) -> Float64

Compute the pair counting recall between two clustering of the same data points.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

Returns the value of the pair counting recall.

# References
> Pfitzner, Darius, Richard Leibbrandt, and David Powers. (2009).
> *Characterization and evaluation of similarity measures for pairs of clusterings.*
> Knowledge and Information Systems: 361-394.
"""
function pair_recall(a, b)
    tp,fp,fn,tn = _pair_confusion_matrix(a,b)
    return tp/(tp+fn)
end

"""
    fmeasure(a, b) -> Float64

Compute the pair counting fmeasure between two clustering of the same data points.

`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).

Returns the value of the pair counting recall.

# References
> Pfitzner, Darius, Richard Leibbrandt, and David Powers. (2009).
> *Characterization and evaluation of similarity measures for pairs of clusterings.*
> Knowledge and Information Systems: 361-394.
"""
function fmeasure(a, b)
    p = pair_precision(a,b)
    r = pair_recall(a,b)
    return (2*p*r)/(p+r)
end