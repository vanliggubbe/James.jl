# weight functions for different scenarios
function _aaa_weights(
    xs :: AbstractVector{<: Real},
    fs :: AbstractVector{<: Number},
    f̄s :: AbstractVector{<: Number},
    js, αs, w_symm :: Number
)
    _, _, V = svd!(
        [
            (
                fs[α] / (xs[j] - xs[α]) +
                f̄s[α] / (xs[j] + xs[α]) * w_symm -
                fs[j] / (xs[j] - xs[α]) -
                f̄s[j] / (xs[j] + xs[α]) * w_symm
            ) for j in js, α in αs
        ]
    )
    return V[:, end], w_symm * V[:, end]
end

function _aaa_weights(
    xs :: AbstractVector{<: Real},
    fs :: AbstractVector{<: AbstractArray},
    f̄s :: AbstractVector{<: AbstractArray},
    js, αs, w_symm :: Number
)
    T = promote_type(eltype(xs), eltype(eltype(fs)), eltype(eltype(f̄s)), typeof(w_symm))
    A = BlockedArray{T}(undef, [length(fs[j]) for j in js], [length(αs)])
    for (i, j) in enumerate(js)
        A[Block(i), Block(1)] .= reduce(
            hcat,
            (
                vec(fs[α]) / (xs[j] - xs[α]) +
                vec(f̄s[α]) / (xs[j] + xs[α]) * w_symm -
                vec(fs[j]) / (xs[j] - xs[α]) -
                vec(f̄s[j]) / (xs[j] + xs[α]) * w_symm
            ) for α in αs
        )
    end
    #_, _, V = svd((eltype(A) <: Complex) ? [real(A); imag(A)] : A)
    _, _, V = svd(A)
    return V[:, end], w_symm * V[:, end]
end

function _aaa_weights(
    xs :: AbstractVector{<: Real},
    fs :: AbstractVector{<: Number},
    f̄s :: AbstractVector{<: Number},
    js, αs, w_symm :: AbstractMatrix{<: Real}
)
    @argcheck size(w_symm) == (2, 2)
    T = promote_type(eltype(xs), real(eltype(fs)), real(eltype(f̄s)), eltype(w_symm))
    A = BlockArray{T}(undef_blocks, [2 * length(js)], fill(2, length(αs)))
    for (i, α) in enumerate(αs)
        A[Block(1), Block(i)] = reinterpret(
            T, hcat(
                [
                    fs[α] / (xs[j] - xs[α]) + 
                    f̄s[α] / (xs[j] + xs[α]) * (w_symm[1, 1] + w_symm[2, 1] * im) -
                    fs[j] / (xs[j] - xs[α]) -
                    f̄s[j] / (xs[j] + xs[α]) * (w_symm[1, 1] + w_symm[2, 1] * im)
                    for j in js
                ],
                [
                    fs[α] / (xs[j] - xs[α]) * im + 
                    f̄s[α] / (xs[j] + xs[α]) * (w_symm[1, 2] + w_symm[2, 2] * im) -
                    fs[j] / (xs[j] - xs[α]) * im -
                    f̄s[j] / (xs[j] + xs[α]) * (w_symm[1, 2] + w_symm[2, 2] * im)
                    for j in js
                ]
            )
        )
    end
    _, __, V = svd(A)
    ws = reinterpret(complex(eltype(V)), V[:, end])
    return ws, [[1; -im]' * w_symm * [real(w); imag(w)] for w in ws]
end

function _aaa_weights(
    xs :: AbstractVector{<: Real},
    fs :: AbstractVector{<: AbstractArray},
    f̄s :: AbstractVector{<: AbstractArray},
    js, αs, w_symm :: AbstractMatrix{<: Real}
)
    @argcheck size(w_symm) == (2, 2)
    T = promote_type(eltype(xs), real(eltype(eltype(fs))), real(eltype(eltype(f̄s))), eltype(w_symm))
    A = BlockArray{T}(undef_blocks, [2 * length(fs[j]) for j in js], fill(2, length(αs)))
    for (i, α) in enumerate(αs)
        for (k, j) in enumerate(js)
            A[Block(k), Block(i)] = reinterpret(
                T, hcat(
                    vec(fs[α]) / (xs[j] - xs[α]) + 
                    vec(f̄s[α]) / (xs[j] + xs[α]) * (w_symm[1, 1] + w_symm[2, 1] * im) -
                    vec(fs[j]) / (xs[j] - xs[α]) -
                    vec(f̄s[j]) / (xs[j] + xs[α]) * (w_symm[1, 1] + w_symm[2, 1] * im),
                    vec(fs[α]) / (xs[j] - xs[α]) * im + 
                    vec(f̄s[α]) / (xs[j] + xs[α]) * (w_symm[1, 2] + w_symm[2, 2] * im) -
                    vec(fs[j]) / (xs[j] - xs[α]) * im -
                    vec(f̄s[j]) / (xs[j] + xs[α]) * (w_symm[1, 2] + w_symm[2, 2] * im),
                )
            )
        end
    end
    _, __, V = svd(A)
    ws = reinterpret(complex(eltype(V)), V[:, end])
    return ws, [[1; -im]' * w_symm * [real(w); imag(w)] for w in ws]
end

@generated function aaa_weights(xs, fs, f̄s, js, αs, w_symm)
    FTYPE = promote_type(deep_eltype(fs), deep_eltype(f̄s))
    if FTYPE <: Real
        return :(_aaa_weights(xs, fs, f̄s, js, αs, first(w_symm)))
    else
        return :(_aaa_weights(xs, fs, f̄s, js, αs, w_symm))
    end
end


function aaa_symm(
    f :: Function,
    Λ :: Real;
    finite :: Bool = false,
    ε :: Real = 1e-8,
    n_iter :: Int = 40,
    n_split :: Function = ConstFun(10),
    f_symm :: Function = conj,
    w_symm = 1,
    point_norm :: Function = norm,
    norm_weight :: Function = ConstFun(1),
)
    @argcheck ispos(Λ)
    @argcheck ispos(ε)
    @argcheck ispos(n_iter)
    @argcheck isone(w_symm * w_symm)
    
    φ_max = finite ? π / 4 : π / 2
    xs = collect(LinRange(0.0, φ_max, ensure(n_split(0), ispos) + 2)[begin + 1 : end - 1])
    zs = Λ * tan.(xs)
    fs = f.(zs)
    f̄s = [f_symm(copy(f)) for f in fs]

    gs = [2 * f for f in fs]

    js = collect(1 : length(xs))    # indices of the probe points
    αs = eltype(js)[]               # indices of the support points

    local ws, w̄s                    # weights
    local er
    for it in 1 : n_iter
        # find the new support point
        er, j = let
            fun(j) = ensure(point_norm(fs[j] - gs[j]), isnneg) * ensure(norm_weight(zs[j]), isnneg)
            findmax(fun, js)
        end
        if er < ε
            break
        end
        jj = js[j]
            
        # add new support points
        # find left and right points closest to the support point to be added
        x = xs[jj]
        yl, yr = let left = filter(<(x), xs[αs]), right = filter(>(x), xs[αs])
            maximum(left; init = zero(eltype(xs))), minimum(right; init = φ_max)
        end
        push!(αs, jj)

        # delete all the probe points between supports
        cur = 0
        for i in eachindex(js)
            if xs[js[i]] < yl || xs[js[i]] > yr
                cur += 1
                js[cur] = js[i]
            end
        end
        resize!(js, cur)

        # split the intervals, add more points
        nn = ensure(n_split(it), ispos)
        for new_xs in [
                LinRange(yl, x, nn + 2)[begin + 1 : end - 1],
                LinRange(x, yr, nn + 2)[begin + 1 : end - 1]
        ]
            new_zs = Λ * tan.(new_xs)
            new_fs = map(f, new_zs)
            append!(js, length(xs) .+ (1 : nn))
            append!(xs, new_xs)
            append!(zs, new_zs)
            append!(fs, new_fs)
            append!(gs, copy(f) for f in new_fs)
            append!(f̄s, f_symm(copy(f)) for f in new_fs)
        end

        ws, w̄s = aaa_weights(zs, fs, f̄s, js, αs, w_symm)

        for j in js
            gs[j] = sum(
                ws[i] * fs[α] / (zs[j] - zs[α]) + w̄s[i] * f̄s[α] / (zs[j] + zs[α])
                for (i, α) in enumerate(αs)
            ) / sum(
                ws[i] / (zs[j] - zs[α]) + w̄s[i] / (zs[j] + zs[α])
                for (i, α) in enumerate(αs)
            )
        end
    end
    return zs[αs], ws, fs[αs], er
end
