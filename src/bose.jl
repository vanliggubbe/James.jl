function aaa_bose(
        T,
        Ω = T,
        δ = default_atol(T, Ω),
        atol = default_atol(T, Ω, δ),
        rtol = default_rtol(atol, T, Ω, δ);
        aaa_kwargs...
)
    bose(x) = (coth(x / (2 * T)) - 2 * T / x) / x + δ
    return aaa_symm(bose, Ω; atol = δ, f_symm = identity, w_symm = -1, aaa_kwargs...)
end

# 1 / a - 1 / b = (a - b) / (a * b)
# cosh(x) / sinh(x) - 1 / x = [cosh(x) * x  - sinh(x)] / (x * sinh(x))
# = [x * (1 + x ^ 2 / 2! + x ^ 4 / 4!) - x - x ^ 3 / 3! - x ^ 5 / 5!]
# / [x ^ 2 + x ^ 4 / 3! + x ^ 6 / 5!] 
# = [x / 2 - x / 6 + x ^ 4 / 
#

function _bose(ω :: Real, p :: Tuple{Real, Real}) 
    T, Ω = p
    RTYPE = float(promote_type(typeof(ω), typeof(T), typeof(Ω)))
    iszero(ω) && return convert(RTYPE, 2 * T)

    weight = one(RTYPE) + (ω / Ω) ^ 2
    iszero(T) && return convert(RTYPE, abs(ω) / weight)
    ω * coth(ω / (2 * T)) / weight
end

_inversion(x) = (one(x) - x) / x

function bose_factor(
        T :: Real,
        Ω :: Real = max(4 * T, 2 * one(T)),
        Λ :: Real = max(2 * T, one(T));
        δ :: Real = sqrt(eps(float(promote_type(typeof(T), typeof(Ω), typeof(Λ))))),
        tol = δ / 64,
        aaa_kwargs...
)
    @argcheck ispos(Λ)
    @argcheck ispos(Ω)
    @argcheck isnneg(T)
    @argcheck ispos(δ)
    # get AAA approximation for coth(x / 2T)
    
    xs, ws, fs = scalar_aaa(
        Base.Fix2(_bose, (T, Ω)) ∘
        Base.Fix2(*, Λ) ∘
        sqrt ∘ _inversion,
        zero(δ), one(δ), false, false;
        tol, aaa_kwargs...
    )

    # Find poles of the approximation
    poles = map(
        Base.Fix2(*, Λ) ∘ sqrt ∘ complex,
        barycentric_poles(_inversion.(xs), ws ./ xs)
    )
    if any(isreal, poles)
        @error "Roots on the real axis: $(real(filter(isreal, poles)))"
        error("Failed to construct rational approximation of n_bose(-ω) * ω")
    end
    poles .*= -sign.(imag(poles))

    # Find roots
    roots, constant = let ωs = Λ * sqrt.(_inversion.(xs))
        a₀ = -xsum(ws .* fs ./ xs) / Ω ^ 2
        ps = -ws ./ xs
        q₀s = -ws .* fs ./ xs .* (1 .+ (ωs / Ω) .^ 2)

        δa = -xsum(ws ./ xs) / Ω ^ 2 * δ
        δqs = -ws ./ xs .* (1 .+ (ωs / Ω) .^ 2) * δ
        i = 1
        local roots
        while true
            a = a₀ + i * δa
            qs = q₀s + i * δqs

            A = [
                -a kron(ones(length(ps)), [-1, 0])';
                (kron(ps, [1, 0]) + kron(qs ./ ωs, [0, 1])) kron(Diagonal(ωs), [0 1; 1 0]);
            ]

            B = Matrix(one(eltype(A)) * I, size(A))
            B[1, 1] = 0

            roots = filter(isfinite, eigvals(A, B))

            # if no real roots, then good to go
            if !any(isreal, roots)
                break
            end
            # otherwise trying to increase by small number
            i += 1
        end
        constant = sqrt((xsum(fs .* ws ./ xs) / xsum(ws ./ xs) + i * δ) / 2) / Ω 

        filter(isneg ∘ imag, roots), constant
    end

    # check for spurious roots
    # TODO make it cleaner
    idx = findall(Base.Fix2(isimag, Approx()), roots) 
    if !isempty(idx)
        # poles to delete and roots to delete
        ptd = eltype(idx)[]
        rtd = eltype(idx)[]
        for i in idx
            j = findall(≈(im * imag(roots[i])), poles)
            @check length(j) < 2
            if !isempty(j)
                push!(ptd, first(j))
                push!(rtd, i)
            end
        end
        sort!(ptd; by = (-))
        sort!(rtd; by = (-))
        for i in ptd
            deleteat!(poles, i)
        end
        for i in rtd
            deleteat!(roots, i)
        end
    end

    @check length(poles) + 1 == length(roots)
    @check isimag(poles, Approx(poles))

    regular = constant * [xsum([poles; -roots]), one(eltype(roots))]
    residues = [
        exp(
            xsum([
                [log(p - r) for r in roots];
                [(j == i ? zero(q) : -log(p - q)) for (j, q) in enumerate(poles)]
            ])
        ) for (i, p) in enumerate(poles)
    ] * constant
    left = map(sqrt ∘ abs, residues)
    return RationalPencil(
        regular, -im,
        left, Diagonal(real(im * poles)), residues ./ left
    )
end
