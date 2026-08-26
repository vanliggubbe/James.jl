function aaa_bose(T, ε, δ = 0, Ω = T; aaa_kwargs...)
    bose(x) = (coth(x / (2 * T)) - 2 * T / x) / x + δ
    return aaa_symm(bose, Ω; ε = ε, f_symm = identity, w_symm = -1, aaa_kwargs...)
end

function bose_factor(
        temperature;
        Ω :: Real = temperature,
        ε :: Real = sqrt(eps(Float64)),
        δ :: Real = (ε / 2),
        aaa_kwargs = ()
)
    @argcheck ispos(δ)
    # get AAA approximation for coth(x / 2T)
    xs, ws, fs = aaa_bose(temperature, ε / 2, δ, Ω; aaa_kwargs...)
    #fs .+= δ

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

    # find roots of the numerator
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
            @error "Roots on the real axis: $(real(filter(isreal, roots)))"
            error("Rational approximation for n_bose(-ω) * ω is not positive. Try increasing δ.")
        end
        filter(isneg ∘ imag, filter(isfinite, eigvals(A, B)))
    end
    @assert length(poles) + 1 == length(roots)
    return (
        sqrt(constant / 2) * 
        FactoredPolynomial(Dict(root => 1 for root in roots)) //
        FactoredPolynomial(Dict(pole => 1 for pole in poles))
    )
end
