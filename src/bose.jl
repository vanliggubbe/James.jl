function aaa_odd(
    a :: Real, f :: Function;
    ε :: Real = 1e-12, n_iter :: Int = 40, split :: Int = 10
)
    xs = collect(LinRange(0.0, a, split + 2)[2 : end])
    fs = f.(xs)
    gs = zero(fs)

    XS = eltype(xs)[]
    FS = eltype(fs)[]
    js = collect(1 : length(xs))
    local WS
    for i in 1 : n_iter
        _, j = findmax(i -> norm(fs[i] - gs[i]), js)
        jj = js[j]
        deleteat!(js, j)
        push!(XS, xs[jj])
        push!(FS, fs[jj])
            
        left = xs .< XS[end]
        y = sum(left) == 0 ? 0.0 : maximum(xs[left])
        append!(xs, LinRange(y, XS[end], split + 2)[2 : end - 1])
        append!(fs, f.(xs[end - split + 1 : end]))
        append!(js, length(xs) - split + 1 : length(xs))

        right = xs .> XS[end]
        if sum(right) != 0
            y = minimum(xs[right])
            append!(xs, LinRange(XS[end], y, split + 2)[2 : end - 1])
            append!(fs, f.(xs[end - split + 1 : end]))
            append!(js, length(xs) - split + 1 : length(xs))
        end

        A = [
            (
                (FS[k] - fs[α]) / (xs[α] - XS[k]) +
                (FS[k] + fs[α]) / (xs[α] + XS[k])
            ) for α in js, k in eachindex(XS, FS)
        ]
        _, __, V = svd!(A)
        WS = V[:, end]

        resize!(gs, length(fs))
        gs .= fs
        for α in js
            gs[α] = sum(
                W * F / (xs[α] - X) + W * F / (xs[α] + X)
                for (X, F, W) in zip(XS, FS, WS)
            ) / sum(
                W / (xs[α] - X) - W / (xs[α] + X)
                for (X, W) in zip(XS, WS)
            )
        end
        err = maximum(norm.(fs - gs))
        if err < ε / 2
            break
        end
    end

    #XS = XS[abs.(WS) .> ε]
    #FS = FS[abs.(WS) .> ε]
    #WS = WS[abs.(WS) .> ε]

    # get poles
    B = Matrix(1.0I, length(WS) + 1, length(WS) + 1)
    B[1, 1] = 0.0
    E = zeros(promote_type(eltype(FS), eltype(XS)), size(B))
    E[1, 2 : end] .= WS
    E[2 : end, 1] .= XS
    E[2 : end, 2 : end] = Diagonal(XS .^ 2)
    pol = map(sqrt ∘ complex, filter(isfinite, eigvals(E, B)))
    pol = [pol; -pol]

    # get residues
    num = [
        sum(
            W * F / (p - X) + W * F / (p + X)
            for (X, F, W) in zip(XS, FS, WS)
        )
        for p in pol
    ]
    den = [
        sum(
            -W / (p - X) ^ 2 + W / (p + X) ^ 2
            for (X, W) in zip(XS, WS)
        )
        for p in pol
    ]
    res = num ./ den

    # get linear part
    lin = (transpose(WS) * FS) / (transpose(WS) * XS)
    return lin + ε, pol, res
end

function aaa_bose(T, ε; aaa_kwargs...)
    bose(x) = (coth(x / (2 * T)) - 2 * T / x) / x
    return aaa_symm(bose, T; ε = ε, f_symm = identity, w_symm = -1, aaa_kwargs...)
end

function bose_factor(
        temperature;
        ε :: Real = sqrt(eps(Float64)),
        δ :: Real = (ε / 2),
        aaa_kwargs = ()
)
    # get AAA approximation for coth(x / 2T)
    xs, ws, fs = aaa_bose(temperature, ε / 2; aaa_kwargs...)
    fs .+= δ

    TYPE = promote_type(eltype(xs), eltype(ws), eltype(fs))
    n = length(xs)
    poles, constant = let 
        # find poles
        A = zeros(TYPE, n + 1, n + 1)
        B = zeros(real(TYPE), n + 1, n + 1)
        for (i, (x, w)) in enumerate(zip(xs, ws))
            A[i + 1, i + 1] = x ^ 2
            tmp = sqrt(abs(w) * abs(x))
            A[1, i + 1] = sign(w) * tmp
            A[i + 1, 1] = tmp
            B[i + 1, i + 1] = one(eltype(B))
        end
        fun(x) = flipsign(x, -imag(x))
        zs = map(fun ∘ sqrt ∘ complex, filter(isfinite, eigvals(A, B)))

        # find residues
        # rs = [
        #     sum(w * f * x / (z ^ 2 - x ^ 2) for (x, w, f) in zip(xs, fs, ws)) /
        #     sum(-2 * w * x * z / (z ^ 2 - x ^ 2) ^ 2 for (x, w) in zip(xs, ws))
        #     for z in zs
        # ]
        zs, sum(w * f * x for (x, w, f) in zip(xs, fs, ws)) / sum(w * x for (x, w) in zip(xs, ws))
    end

    roots = let
        A = zeros(TYPE, 2 * n + 1, 2 * n + 1)
        B = zeros(TYPE, 2 * n + 1, 2 * n + 1)
        for (i, (x, w, f)) in enumerate(zip(xs, ws, fs))
            A[i * 2 + 0, i * 2 + 0] = x
            A[i * 2 + 1, i * 2 + 1] = -x
            qwe_p = x + f * x ^ 2 + 2 * temperature
            qwe_m = x - f * x ^ 2 - 2 * temperature
            tmp_p = sqrt(abs(w) * abs(qwe_p))
            tmp_m = sqrt(abs(w) * abs(qwe_m))
            A[1, i * 2 + 0] = sign(w) * tmp_p
            A[1, i * 2 + 1] = sign(w) * tmp_m
            A[i * 2 + 0, 1] = sign(qwe_p) * tmp_p
            A[i * 2 + 1, 1] = sign(qwe_m) * tmp_m
            B[i * 2 + 0, i * 2 + 0] = one(eltype(B))
            B[i * 2 + 1, i * 2 + 1] = one(eltype(B))
        end
        A[1, 1] = 2 * sum(f * w * x for (f, w, x) in zip(fs, ws, xs))

        roots = filter(isfinite, eigvals(A, B))
        if any(isreal, roots)
            error("
        end
        @assert !any(isreal, roots)
        filter(isneg ∘ imag, filter(isfinite, eigvals(A, B)))
    end
    @assert length(poles) + 1 == length(roots)
    return (
        sqrt(constant / 2) * 
        FactoredPolynomial(Dict(root => 1 for root in roots)) //
        FactoredPolynomial(Dict(pole => 1 for pole in poles))
    )
    # find AAA approximation for coth
    #=
    τ, νs, rs = aaa_odd(Λ, x -> coth(x / (2 * T)) - 2 * T / x; ε)
    bose(x) = τ * x + 2 * T / x + sum(rs ./ (x .- νs))

    # find zeros of AAA approximation
    # τ ω + 2T / ω + sum_k r_k / (ω - ν_k) + 1 = 0
    B = Matrix(1.0I, length(νs) + 2, length(νs) + 2)
    B[1, 1] = -τ

    E = zeros(ComplexF64, size(B))
    E[1, 2] = 2 * T
    E[1, 3 : end] .= rs
    E[2 : end, 1] .= 1
    for (i, ν) in enumerate(νs)
        E[i + 2, i + 2] = ν
    end
    E[1, 1] = 1
    roots = filter(isfinite, eigvals(E, B))
    return (
        sqrt(τ / 2) * FactoredPolynomial(Dict(root => 1 for root in roots[imag(roots) .< 0])) //
        FactoredPolynomial(Dict(root => 1 for root in νs[imag(νs) .< 0]))
    )
    =#
end
