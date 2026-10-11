# 1 / a - 1 / b = (a - b) / (a * b)
# cosh(x) / sinh(x) - 1 / x = [cosh(x) * x  - sinh(x)] / (x * sinh(x))
# = [x * (1 + x ^ 2 / 2! + x ^ 4 / 4!) - x - x ^ 3 / 3! - x ^ 5 / 5!]
# / [x ^ 2 + x ^ 4 / 3! + x ^ 6 / 5!] 
# = [x / 2 - x / 6 + x ^ 4 / 
#

function _bose(ω :: Real, p :: Tuple{Real, Real, Real}) 
    T, Ω, δ = p
    RTYPE = float(promote_type(typeof(ω), typeof(T), typeof(Ω)))

    isinf(ω) && return zero(RTYPE)
    iszero(ω) && return convert(RTYPE, 2 * T)

    weight = one(RTYPE) + (ω / Ω) ^ 2
    iszero(T) && return convert(RTYPE, abs(ω) / weight)
    ((ω * coth(ω / (2 * T)) + 0 * δ * ω ^ 2 / Ω ^ 2 / 2) / weight)
end

_inversion(x) = iszero(x) ? convert(typeof(x), Inf) : (one(x) - x) / x

function bose_factor(
        T :: Real,
        Ω :: Real = max(4 * T, 2 * one(T));
        Λ :: Real = Ω,
        δ :: Real = sqrt(eps(float(promote_type(typeof(T), typeof(Ω), typeof(Λ))))),
        tol = δ / 2,
        aaa_kwargs...
)
    @argcheck ispos(Λ)
    @argcheck ispos(Ω)
    @argcheck isnneg(T)
    @argcheck ispos(δ)
    # get AAA approximation for coth(x / 2T)
    
    # convert everything to double floats
    # since it is a very precision sensitive part of the code
    T_128 = convert(Double64, T)
    Ω_128 = convert(Double64, Ω)
    Λ_128 = convert(Double64, Λ)
    δ_128 = convert(Double64, δ)
    tol_128 = convert(Double64, tol)
    xs, ws, fs = scalar_aaa(
        Base.Fix2(_bose, (T_128, Ω_128, δ_128)) ∘
        Base.Fix2(*, Λ_128) ∘
        sqrt ∘ _inversion,
        zero(Double64), one(Double64), true, false;
        force_poles = [Ω == Λ ? Double64(Inf) : (Λ_128 ^ 2 / (Λ_128 ^ 2 - Ω_128 ^ 2))],
        x_map = sqrt,
        tol = tol_128, aaa_kwargs...
    )
    #1 + ω ^ 2 / Ω ^ 2 =
    #1 + Λ ^ 2 / Ω ^ 2 * (1 - x) / x
    #(Ω ^ 2 * x + Λ ^ 2 * (1 - x)) / x

    maxdev, mindev = let
        p = sortperm(xs)
        curx = zero(eltype(fs))
        curn = zero(eltype(fs))
        for (a, b) in zip(xs[p], [xs[p[2 : end]]; one(eltype(xs))])
            l, r = a, b
            if isone(b)
                curx = max(curx, sum_kbn(fs .* ws ./ (b .- xs)) / sum_kbn(ws ./ (b .- xs)) - 2 * T_128)
                curn = min(curn, sum_kbn(fs .* ws ./ (b .- xs)) / sum_kbn(ws ./ (b .- xs)) - 2 * T_128)
            end
            for _ in 1 : 50
                xl, xr = (2 * l + r) / 3, (l + 2 * r) / 3
                fl = sum_kbn(fs .* ws ./ (xl .- xs)) / sum_kbn(ws ./ (xl .- xs)) - _bose(Λ_128 * sqrt(_inversion(xl)), (T_128, Ω_128, δ_128))
                fr = sum_kbn(fs .* ws ./ (xr .- xs)) / sum_kbn(ws ./ (xr .- xs)) - _bose(Λ_128 * sqrt(_inversion(xr)), (T_128, Ω_128, δ_128))
                if isnan(fl) || isnan(fr)
                    break
                end
                curx = max(curx, fl)
                curx = max(curx, fr)
                if fl > fr
                    r = xr
                else
                    l = xl
                end
            end
            l, r = a, b
            for _ in 1 : 50
                xl, xr = (2 * l + r) / 3, (l + 2 * r) / 3
                fl = sum_kbn(fs .* ws ./ (xl .- xs)) / sum_kbn(ws ./ (xl .- xs)) - _bose(Λ_128 * sqrt(_inversion(xl)), (T_128, Ω_128, δ_128))
                fr = sum_kbn(fs .* ws ./ (xr .- xs)) / sum_kbn(ws ./ (xr .- xs)) - _bose(Λ_128 * sqrt(_inversion(xr)), (T_128, Ω_128, δ_128))
                if isnan(fl) || isnan(fr)
                    break
                end
                curn = min(curn, fl)
                curn = min(curn, fr)
                if fl < fr
                    r = xr
                else
                    l = xl
                end
            end
        end
        curx, curn
    end
    #println(maxdev, " ", mindev)

    if iszero(first(ws))
        ws = ws[2 : end]
        xs = xs[2 : end]
        fs = fs[2 : end]
    end

    # Find poles of the approximation
    poles = if iszero(first(xs))
        # p = barycentric_poles(xs, ws)
        # poles = map(sqrt ∘ complex ∘ _inversion, p)
        # poles .*= Λ_128
        
        poles = map(
            Base.Fix2(*, Λ_128) ∘ sqrt ∘ complex,
            barycentric_poles(
                _inversion.(xs[2 : end]),
                ws[2 : end] ./ xs[2 : end],
                -first(ws)
            )
        )
        
        if any(isreal, poles)
            @error "Roots on the real axis: $(real(filter(isreal, poles)))"
            error("Failed to construct rational approximation of n_bose(-ω) * ω")
        end
        -(poles .* sign.(imag(poles)))
    else
        # p = barycentric_poles(xs, ws)
        # poles = map(sqrt ∘ complex ∘ _inversion, p)
        # poles .*= Λ_128

        poles = map(
            Base.Fix2(*, Λ_128) ∘ sqrt ∘ complex,
            barycentric_poles(_inversion.(xs), ws ./ xs)
        )

        if any(isreal, poles)
            @error "Roots on the real axis: $(real(filter(isreal, poles)))"
            error("Failed to construct rational approximation of n_bose(-ω) * ω")
        end
        -(poles .* sign.(imag(poles)))
    end

    # Find roots
    right = abs(mindev)
    roots, constant = if iszero(first(xs))
        ws .*= sign(first(ws))
        w₀ = first(ws)
        f₀ = first(fs)
        ws = ws[2 : end]
        fs = fs[2 : end]
        xs = xs[2 : end]
        
        ωs = Λ_128 * sqrt.(_inversion.(xs))
        #a₀ = -sum(ws .* fs ./ xs) / Ω ^ 2 + w₀ * f₀ / Λ ^ 2
        a₀ = sum_kbn([w₀ * f₀ / Λ_128 ^ 2; -ws .* fs ./ xs / Ω_128 ^ 2])
        b  = w₀ / Λ_128 ^ 2
        c₀ = f₀ * w₀ / Λ_128 ^ 2 / Ω_128 ^ 2
        ps = -ws ./ xs
        q₀s = -ws .* fs ./ xs .* (one(Double64) .+ (ωs / Ω_128) .^ 2)

        #δa = (-sum(ws ./ xs) / Ω ^ 2 + w₀ / Λ ^ 2)
        δa = sum_kbn([w₀ / Λ_128 ^ 2; -ws ./ xs / Ω_128 ^ 2])
        δc = w₀ / Λ_128 ^ 2 / Ω_128 ^ 2
        δqs = -ws ./ xs .* (one(Double64) .+ (ωs / Ω_128) .^ 2)

        left = zero(Double64)
        i = 0

        local roots
        while i <= 40
            shift = iszero(i) ? right : (left + right) / 2
            a = a₀ + δa * shift
            c = c₀ + δc * shift
            qs = q₀s + δqs * shift

            companion = [
                zero(Double64)     -a;
                one(Double64)      -b / sqrt(c)
            ]

            ns = sqrt.(hypot.(ps, qs ./ ωs))
            A = [
                companion kron(ns', [-1 0; 0 0]);
                (kron(ps ./ ns, [0 1; 0 0]) + kron(qs ./ ωs ./ ns, [0 0; 0 1])) kron(Diagonal(ωs), [0 1; 1 0]);
            ]

            B = Diagonal([sqrt(c); sqrt(c); ones(size(A, 1) - 2)])
            tmp = eigvals(sqrt(B) \ (A / sqrt(B)))

            if i != 0
                i += 1
            end

            # if no real roots, then good to go
            if !any(isreal, tmp)
                roots = tmp
                if i == 0
                    i = 1
                else
                    right = shift
                end
            else
                if i == 0
                    #display(filter(isreal, tmp))
                    right *= 2
                else
                    left = shift
                end
            end
        end
        constant = sqrt(right / 2) / Ω_128
        filter(isneg ∘ imag, roots), constant
    else
        ωs = Λ_128 * sqrt.(_inversion.(xs))
        a₀ = -sum_kbn(ws .* fs ./ xs) / Ω_128 ^ 2
        ps = -ws ./ xs
        q₀s = -ws .* fs ./ xs .* (one(Double64) .+ (ωs / Ω_128) .^ 2)

        δa = -sum_kbn(ws ./ xs) / Ω_128 ^ 2
        δqs = -ws ./ xs .* (one(Double64) .+ (ωs / Ω_128) .^ 2)

        left = zero(Double64)
        i = 0

        local roots
        while i <= 40
            shift = iszero(i) ? right : (left + right) / 2
            a = a₀ + shift * δa
            qs = q₀s + shift * δqs

            A = [
                -a kron(ones(length(ps)), [-1, 0])';
                (kron(ps, [1, 0]) + kron(qs ./ ωs, [0, 1])) kron(Diagonal(ωs), [0 1; 1 0]);
            ]

            B = Matrix(one(eltype(A)) * I, size(A))
            B[1, 1] = zero(eltype(A))

            f = schur(A, B)
            tmp = filter((!) ∘ isnan, filter(isfinite, f.α ./ f.β))
            if i != 0
                i += 1
            end

            # if no real roots, then good to go
            if !any(isreal, tmp)
                roots = tmp
                if i == 0
                    i = 1
                else
                    right = shift
                end
            else
                if i == 0
                    right *= 2
                else
                    left = shift
                end
            end
        end
        constant = sqrt((sum_kbn(fs .* ws ./ xs) / sum_kbn(ws ./ xs) + right) / 2) / Ω_128
        filter(isneg ∘ imag, roots), constant
    end

    @check length(poles) + 1 == length(roots)
    @check isimag(poles, Approx(poles))

    # delete -im Ω from the poles and roots
    deleteat!(poles, last(findmin(abs, poles .+ im * Ω)))
    deleteat!(roots, last(findmin(abs, roots .+ im * Ω)))

    regular = constant * [sum([poles; -roots]), one(eltype(roots))]
    residues = [
        exp(
            sum_kbn([
                [log(p - r) for r in roots];
                [(j == i ? zero(q) : -log(p - q)) for (j, q) in enumerate(poles)]
            ])
        ) for (i, p) in enumerate(poles)
    ] * constant
    left = map(sqrt ∘ abs, residues)

    RTYPE = float(promote_type(typeof(T), typeof(Ω), typeof(Λ), typeof(δ)))
    f = RationalPencil(
        convert.(complex(RTYPE), regular), -im,
        convert.(RTYPE, left), 
        Diagonal(convert.(RTYPE,real(im * poles))), 
        convert.(complex(RTYPE), residues ./ left)
    )
    #println(maxdev, " ", mindev, " ", right)
    return f, convert(RTYPE, right + maxdev)
end
