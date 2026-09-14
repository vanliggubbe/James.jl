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
_bose(x) = (
    abs(x) < eps(eltype(x)) ^ 0.25 ?
    (one(x) + x ^ 2 / 10) / (one(x) + x ^ 2 / 6) / 3 :
    (coth(x) - inv(x)) / x
)

_inversion(x) = (one(x) - x) / x

function _coth_approx(δ :: Real, Λ :: Real = one(δ); aaa_kwargs...)
    end

function bose_factor(
        temperature :: Real,
        δ :: Real = sqrt(eps(float(typeof(temperature)))),
        Ω :: Real = temperature * 4;
        tol = δ / 64,
        aaa_kwargs...
)
    @argcheck ispos(temperature)
    @argcheck ispos(δ)
    # get AAA approximation for coth(x / 2T)
    
    Λ = Ω / (2 * temperature)
    Λ² = Λ ^ 2
    xs, ws, fs = scalar_aaa(
        _bose ∘ sqrt ∘ Base.Fix2(*, Λ²) ∘ _inversion,
        zero(δ), one(δ), false, false;
        tol,
        aaa_kwargs...
    )
    
    poles, constant = let n = length(xs)
        A = zeros(promote_type(eltype(xs), eltype(ws)), n + 1, n + 1)
        A[2 : end, 1] .= map(sqrt ∘ abs, ws)
        A[1, 2 : end] .= A[2 : end, 1] .* sign.(ws)
        for (i, x) in enumerate(xs)
            A[i + 1, i + 1] = x
        end

        B = zeros(eltype(A), n + 1, n + 1)
        B[2 : end, 2 : end] .= I(n)

        map(
            (-) ∘ sqrt ∘ complex ∘ Base.Fix2(*, Λ²) ∘ _inversion,
            filter(isfinite, eigvals(A, B))
        ), xsum(fs .* ws ./ xs) / xsum(ws ./ xs) + δ
    end
    fs .+= δ

    roots = let zs = Λ * sqrt.((1 .- xs) ./ xs)
        a = -xsum(ws .* fs ./ xs)

        ps = -ws ./ xs
        qs = -ws ./ xs .* (1 .+ fs .* (zs .^ 2))

        A = [
            -a kron(ones(length(ps)), [-1, 0])';
            (kron(ps, [1, 0]) + kron(qs ./ zs, [0, 1])) kron(Diagonal(zs), [0 1; 1 0]);
        ]
        B = Matrix(one(eltype(A)) * I, size(A))

        B[1, 1] = 0
        roots = filter(isfinite, eigvals(A, B))
        if any(isreal, roots)
            @error "Roots on the real axis: $(real(filter(isreal, roots)))"
            error("Rational approximation for n_bose(-ω) * ω is not positive. Try increasing `δ` while keeping `tol`.")
        end
        filter(isnpos ∘ imag, roots)
    end
    @check length(poles) + 1 == length(roots)
    @check isreal(im * poles, Approx(poles))

    roots .*= 2 * temperature
    poles .*= 2 * temperature
    constant /= 4 * temperature

    regular = sqrt(constant) * [xsum([poles; -roots]), one(eltype(roots))]
    residues = [
        exp(
            xsum([
                [log(p - r) for r in roots];
                [(j == i ? zero(q) : -log(p - q)) for (j, q) in enumerate(poles)]
            ])
        ) for (i, p) in enumerate(poles)
    ] * sqrt(constant)
    left = map(sqrt ∘ abs, residues)
    return RationalPencil(
        regular, -im,
        left, Diagonal(real(im * poles)), residues ./ left
    )
end
