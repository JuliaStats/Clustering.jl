# Common utilities

##### update minimum value

function updatemin!(r::AbstractArray, x::AbstractArray)
    n = length(r)
    length(x) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    @inbounds for i = 1:n
        xi = x[i]
        if xi < r[i]
            r[i] = xi
        end
    end
    return r
end


##### Column accumulation #####

function accumulate_cols!(
    r::AbstractMatrix,    # destination
    rw::AbstractArray,    # accumulated weights
    x::AbstractMatrix,    # source 
    c::AbstractArray)     # assignments (labels)

    d = size(r, 1)
    K = size(r, 2)
    n = size(x, 2)

    if !(d == size(x, 1) && K == length(rw) && n == length(c))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    for j = 1 : n
        @inbounds cj = c[j]
        1 <= cj <= K || error("assignment out of boundary.")

        rj = view(r, :, cj)
        xj = view(x, :, j)
        if rw[cj] > 0
            add!(rj, xj)
        else
            copy!(rj, xj)
        end
        rw[cj] += 1
    end
end

function accumulate_cols!(
    r::AbstractMatrix,    # destination
    rw::AbstractArray,    # accumulated weights
    x::AbstractMatrix,    # source 
    c::AbstractArray,     # assignments (labels)
    w::AbstractArray)     # column weights

    d = size(r, 1)
    K = size(r, 2)
    n = size(x, 2)

    if !(d == size(x, 1) && K == length(rw) && n == length(c))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    for j = 1 : n
        @inbounds wj = w[j]

        if wj > 0
            @inbounds cj = c[j]
            1 <= cj <= K || error("assignment out of boundary.")

            rj = view(r, :, cj)
            xj = view(x, :, j)
            if rw[cj] > 0
                fma!(rj, xj, wj)
            else
                map!(Multiply(), rj, xj, wj)
            end
            rw[cj] += wj
        end
    end
end

function accumulate_cols_u!(
    r::AbstractMatrix,    # destination
    rw::AbstractArray,    # accumulated weights
    x::AbstractMatrix,    # source 
    c::AbstractArray,     # assignments (labels)
    u::AbstractArray{Bool}) # update control     

    d = size(r, 1)
    K = size(r, 2)
    n = size(x, 2)

    if !(d == size(x, 1) && K == length(rw) && n == length(c))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    for j = 1 : n
        @inbounds cj = c[j]
        1 <= cj <= K || error("assignment out of boundary.")

        if u[cj]
            rj = view(r, :, cj)
            xj = view(x, :, j)
            if rw[cj] > 0
                for i = 1:d
                    @inbounds rj[i] += xj[i]
                end
            else
                copy!(rj, xj)
            end
            rw[cj] += 1
        end
    end
end

function accumulate_cols_u!(
    r::AbstractMatrix,    # destination
    rw::AbstractArray,    # accumulated weights
    x::AbstractMatrix,    # source 
    c::AbstractArray,     # assignments (labels)
    w::AbstractArray,     # column weights
    u::AbstractArray{Bool})  # update control

    d = size(r, 1)
    K = size(r, 2)
    n = size(x, 2)

    if !(d == size(x, 1) && K == length(rw) && n == length(c))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    for j = 1 : n
        @inbounds wj = w[j]

        if wj > 0
            @inbounds cj = c[j]
            1 <= cj <= K || error("assignment out of boundary.")
            
            if u[cj]
                rj = view(r, :, cj)
                xj = view(x, :, j)
                if rw[cj] > 0
                    for i = 1:d
                        @inbounds rj[i] += xj[i] * wj
                    end
                else
                    for i = 1:d
                        @inbounds rj[i] = xj[i] * wj
                    end
                end
                rw[cj] += wj
            end
        end
    end
end



