# weight functions for different scenarios
function _aaa_weights(
    xs :: AbstractVector{<: Real},
    fs :: AbstractVector{<: Number},
    f̄s :: AbstractVector{<: Number},
    js, αs, w_symm :: Number
)
    _, _, V = svd(
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
    atol :: Real = default_atol(Λ),
    rtol :: Real = default_rtol(atol, Λ),
    n_iter :: Int = 40,
    n_split :: Function = Returns(10),
    f_symm :: Function = conj,
    w_symm = 1,
    point_norm :: Function = frnorm,
    norm_weight :: Function = Returns(1),
)
    @argcheck ispos(Λ)
    @argcheck isnneg(atol)
    @argcheck isnneg(rtol)
    @argcheck ispos(n_iter)
    @argcheck isone(w_symm * w_symm)
    
    RTYPE = float(typeof(Λ))
    φ_max = finite ? π / RTYPE(4) : π / RTYPE(2)
    xs = collect(LinRange(zero(RTYPE), φ_max, ensure(n_split(0), ispos) + 2)[begin + 1 : end - 1])
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
        nrm = maximum(point_norm(f) * norm_weight(z) for (f, z) in zip(fs, zs))
        if er < max(atol, rtol * nrm)
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
            gs[j] = sum_kbn(
                [
                    ws[i] * fs[α] / (zs[j] - zs[α]) + w̄s[i] * f̄s[α] / (zs[j] + zs[α])
                    for (i, α) in enumerate(αs)
                ]
            ) / sum_kbn(
                [
                    ws[i] / (zs[j] - zs[α]) + w̄s[i] / (zs[j] + zs[α])
                    for (i, α) in enumerate(αs)
                ]
            )
        end
    end
    return zs[αs], ws, fs[αs], er
end

function barycentric_poles(xs, ws, c = zero(promote_type(eltype(xs), eltype(ws))))
    n = length(xs)

    A = zeros(promote_type(eltype(xs), eltype(ws)), n + 1, n + 1)
    A[2 : end, 1] .= map(sqrt ∘ abs, ws)
    A[1, 2 : end] .= A[2 : end, 1] .* phase_factor.(ws)
    A[1, 1] = c
    for (i, x) in enumerate(xs)
        A[i + 1, i + 1] = x
    end

    # calculate generalized eigenvalues
    # using generalized Schur decomposition
    B = Matrix(one(eltype(A)) * I, n + 1, n + 1)
    B[1, 1] = zero(eltype(B))
    f = schur(A, B)

    return filter((!) ∘ isnan, filter(isfinite, f.α ./ f.β))
end

function aaa_circle(
    f :: Function;
    tol = 1e-9,
    n_split :: Function = Base.Fix1(max, 3) ∘ Base.Fix1(-, 40),
    n_iter :: Int = 100,
)
    RTYPE = real(float(typeof(f(-1))))
    xs = collect(LinRange(RTYPE(0), RTYPE(π), n_split(0) + 2))[2 : end - 1]
    zs = expm1.(im * xs)
    fs = f.(zs)
    gs = 2 * fs
    js = collect(1 : length(xs))    # indices of the probe points
    αs = eltype(js)[]               # indices of the support points
    display(fs)

    local ws
    local ws_best
    inside(x) = isone(abs(x), Approx())
    for it in 1 : n_iter
        # check if 
        #poles = it == 1 ? eltype(zs)[] : filter(inside, barycentric_poles(zs[αs], ws))
        jj = if false && !isempty(poles)
            # add the pole with the largest residue
            res = [
                -sum_kbn(fs[αs] .* ws ./ (p .- zs[αs])) /
                sum_kbn(ws ./ (p .- zs[αs]) .^ 2)
                for p in poles
            ]
            _, k = findmax(abs, res)
            push!(zs, poles[k])
            push!(xs, real(-im * log(zs[end])))
            push!(fs, f(zs[end]))
            push!(js, length(zs))
            js[end]
        else
            if it != 1
                ws_best = copy(ws)
            end
            er, j = let
                fun(j) = abs(fs[j] - gs[j])
                findmax(fun, js)
            end
            #@debug "$(xs[js[j]]) $(er)"
            if er < tol
                break
            end
            println(it, " ", xs[js[j]], " ", er)
            js[j]
        end
            
        # add new support points
        # find left and right points closest to the support point to be added
        x = xs[jj]
        yl, yr = let left = filter(<(x), xs[αs]), right = filter(>(x), xs[αs])
            maximum(left; init = RTYPE(0)), minimum(right; init = RTYPE(π))
        end
        push!(αs, jj)

        # delete all the probe points between supports
        cur = 0
        nleft = n_split(it) - 1  # count(j -> yl < xs[j] < x, js)
        nright = n_split(it) - 1 # count(j -> x < xs[j] < yr, js)
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
                collect(LinRange(yl, x, nleft + 3)[begin + 1 : end - 1]),
                collect(LinRange(x, yr, nright + 3)[begin + 1 : end - 1])
        ]
            new_zs = expm1.(im * new_xs)
            new_fs = map(f, new_zs)
            append!(js, length(xs) .+ (1 : nn))
            append!(xs, new_xs)
            append!(zs, new_zs)
            append!(fs, new_fs)
            append!(gs, zero(fs))
        end

        #C = [inv(zs[j] - zs[α]) for j in js, α in αs]
        ws = let A = zeros(RTYPE, 2 * length(js), 2 * length(αs))#[abs(weight(xs[j])) * (fs[j] - fs[α]) for j in js, α in αs] .* C
            for (r, j) in enumerate(js)
                for (c, α) in enumerate(αs)
                    f1 = fs[α] - fs[j]
                    f2 = conj(fs[α]) - fs[j]
                    z1 = zs[j] - zs[α]
                    z2 = zs[j] - conj(zs[α])
                    if isnan(f1) || isnan(f2)
                        println(fs[α], " ", zs[α])
                        println(fs[j], " ", zs[j])
                        error()
                    end
                    r1 = f1 / z1
                    r2 = f2 / z2
                    A[2 * r .+ [-1, 0], 2 * c .+ [-1, 0]] .= [
                        (real(r1) + real(r2))   (-imag(r1) + imag(r2));
                        (imag(r1) + imag(r2))   (real(r1) - real(r2))
                    ]
                end
            end
            _, _, V = svd(A)
            V[1 : 2 : end, end] + im * V[2 : 2 : end, end]
        end

        for j in js
            gs[j] = sum_kbn(
                [
                    [fs[α] * w / (zs[j] - zs[α]) for (w, α) in zip(ws, αs)];
                    [conj(fs[α] * w) / (zs[j] - conj(zs[α])) for (w, α) in zip(ws, αs)]
                ]
            ) / sum_kbn(
                [
                    [w / (zs[j] - zs[α]) for (w, α) in zip(ws, αs)];
                    [conj(w) / (zs[j] - conj(zs[α])) for (w, α) in zip(ws, αs)]
                ]
            )
        end
        #gs[js] .= xmul(C, (ws .* fs[αs])) ./ xmul(C, ws)
    end
    αs = αs[1 : length(ws_best)]
    zs[αs], ws_best, fs[αs]
end

function scalar_aaa(
    f :: Function,
    a :: Real,
    b :: Real,
    include_a :: Bool = true,
    include_b :: Bool = true;
    force_poles = nothing,
    tol = 1000 * eps(promote_type(typeof(a), typeof(b))),
    n_split :: Function = Base.Fix1(max, 3) ∘ Base.Fix1(-, 30) ∘ identity,
    n_iter :: Int = 100,
    weight :: Function = Returns(1),
    x_map :: Function = identity
)
    @argcheck ispos(tol)
    @argcheck ispos(n_iter)

    xs = collect(LinRange(x_map(a), x_map(b), n_split(0) + 2)) .|> inverse(x_map)
    fs = f.(xs)
    gs = 2 * fs
    fpoles = if isnothing(force_poles)
        eltype(fs)[]
    else
        force_poles
    end

    js = collect(2 : length(xs) - 1)    # indices of the probe points
    αs = eltype(js)[]                   # indices of the support points
    include_a && push!(αs, 1)
    include_b && push!(αs, length(xs))

    inside(x) = a < x < b

    local ws
    local ws_best
    for it in 1 : n_iter
        # check if 
        poles = it == 1 ? eltype(xs)[] : filter(inside, real(filter(isreal, barycentric_poles(xs[αs], ws))))
        jj = if !isempty(poles)
            # add the pole with the largest residue
            res = [
                -sum_kbn(fs[αs] .* ws ./ (p .- xs[αs])) /
                sum_kbn(ws ./ (p .- xs[αs]) .^ 2)
                for p in poles
            ]
            _, k = findmax(abs, res .* weight.(poles))
            push!(xs, poles[k])
            push!(fs, f(xs[end]))
            push!(js, length(xs))
            js[end]
        else
            if it != 1
                ws_best = copy(ws)
            end
            er, j = let
                fun(j) = abs(fs[j] - gs[j]) * abs(weight(xs[j]))
                findmax(fun, js)
            end
            #@debug "$(xs[js[j]]) $(er)"
            if er < tol
                break
            end
            js[j]
        end
            
        # add new support points
        # find left and right points closest to the support point to be added
        x = xs[jj]
        yl, yr = let left = filter(<(x), xs[αs]), right = filter(>(x), xs[αs])
            maximum(left; init = a), minimum(right; init = b)
        end
        push!(αs, jj)

        # delete all the probe points between supports
        cur = 0
        nleft = n_split(it) - 1  # count(j -> yl < xs[j] < x, js)
        nright = n_split(it) - 1 # count(j -> x < xs[j] < yr, js)
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
                collect(LinRange(x_map(yl), x_map(x), nleft + 3)[begin + 1 : end - 1]) .|> inverse(x_map),
                collect(LinRange(x_map(x), x_map(yr), nright + 3)[begin + 1 : end - 1]) .|> inverse(x_map)
        ]
            new_fs = map(f, new_xs)
            append!(js, length(xs) .+ (1 : nn))
            append!(xs, new_xs)
            append!(fs, new_fs)
            append!(gs, zero(fs))
        end


        C = [inv(xs[j] - xs[α]) for j in js, α in αs]
        ws = let A = [abs(weight(xs[j])) * (fs[j] - fs[α]) for j in js, α in αs] .* C
            Q = if length(fpoles) < length(αs)
                N = zeros(eltype(fs), length(αs), length(fpoles)) 
                for (i, p) in enumerate(fpoles)
                    if isfinite(p)
                        N[:, i] .= inv.(p .- xs[αs])
                    else
                        N[:, i] .= one(eltype(N))
                    end
                end
                Q, _ = qr(N)
                Q[:, length(fpoles) + 1 : end]
            else
                I
            end
            _, _, V = svd(A * Q)
            Q * V[:, end]
        end

        gs[js] .= xmul(C, (ws .* fs[αs])) ./ xmul(C, ws)
    end
    αs = αs[1 : length(ws_best)]

    xs[αs], ws_best, fs[αs]
end

# slow but accurate
function xmul(A :: Matrix, b :: Vector)
    c = Vector{promote_type(eltype(A), eltype(b))}(undef, size(A, 1))
    @inbounds for j in eachindex(c)
        c[j] = sum_kbn([p * q for (p, q) in zip((@view A[j, :]), b)])
    end
    return c
end
