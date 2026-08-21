struct BathSpectralDensity{
    T <: Real,
    ML <: AbstractMatrix{T},
    MM <: AbstractMatrix{T}, 
    MR <: AbstractMatrix{T}
}
    L :: ML
    M :: MM
    R :: MR

    function BathSpectralDensity(L, M, R)
        @argcheck ishurwitz(M)
        new{eltype(L), typeof(L), typeof(M), typeof(R)}(L, M, R)
    end
end

spectral_factor(J :: BathSpectralDensity, ω) = J.L' * ((ω * I + im * J.M) \ J.R)

_J_call(J :: BathSpectralDensity, ω) = ω * spectral_factor(J, ω) * (spectral_factor(J, conj(ω)))'
(J :: BathSpectralDensity)(ω) = _J_call(J, ω)
(J :: BathSpectralDensity)(ω :: Real) = Hermitian(_J_call(J, ω))

function BathSpectralDensity(
    J :: Function;
    Ω :: Real = 1.0,
    ε :: Real = 1e-7,
    δ :: Real = (ε / 2),
    aaa_kwargs
)

    bsd(x :: Real) = (@onlyif ensure(J(x), isposdef ∘ Hermitian) / x ispos(x))
    ωs_bsd,  ws_bsd,  fs_bsd  = aaa_symm(bsd, Ω; ε = ε / 2, f_symm = conj, w_symm = [1 0; 0 -1], aaa_kwargs...)

    #=
    # Bath correlation functions
    # S₊(ω) = J(ω) [coth(ω / 2T) + 1] / 2
    # S₋(ω) = J(ω) [coth(ω / 2T) - 1] / 2
    S(ω) = let J = Hermitian(J(ω)), bose = coth(ω / (2 * T))
        @argcheck ispos(ω)
        @argcheck isposdef(J)
        [(J * (bose + 1) / 2) (J * (bose - 1) / 2)]
    end


    # Bath spectral density symmetry: J(-ω) = -[J(ω)]ᵀ
    # S₊(-ω) = -[J(ω)]ᵀ [-coth(ω / 2T) + 1] / 2 = [S₋(ω)]ᵀ
    # S₋(-ω) = -[J(ω)]ᵀ [-coth(ω / 2T) - 1] / 2 = [S₊(ω)]ᵀ
    S_symm(S) = let S₊ = S[:, 1 : size(S, 1)], S₋ = S[:, size(S, 1) .+ (1 : size(S, 1))]
        conj!([S₋ S₊])
    end

    bose(x) = (coth(x / (2 * T)) - 2 * T / x) / x
        println("qwe")

    return ωs_bsd, ws_bsd, fs_bsd

    # get barycentric weights
    ωs, ws, Ss, S̄s = let
        ωs, ws, Ss = aaa_symm(S, Ω; ε = ε / 2, f_symm = S_symm, w_symm = [1 0; 0 -1], aaa_kwargs...)
        ws = real(ws)

        # remove too small weights
        to_delete = findall(<(eps(eltype(ws))) ∘ abs, ws)
        deleteat!(ωs, to_delete)
        deleteat!(ws, to_delete)
        deleteat!(Ss, to_delete)

        S̄s = [conj(S[:, size(S, 1) .+ (1 : size(S, 1))]) for S in Ss]
        Ss = [S[:, 1 : size(S, 1)] for S in Ss]
        ωs, ws, Ss, S̄s
    end
    return ωs, ws, Ss, S̄s

    # find poles of the barycentric approximation
    # since we used symmetrized form, it's more accurate to look directly for poles
    # bit hard to deal if high order poles are present
    poles = let
        TYPE = promote_type(eltype(ωs), eltype(ws))
        n = length(ωs)
        A = zeros(TYPE, n + 1, n + 1)
        B = zeros(real(TYPE), n + 1, n + 1)
        for (j, (x, w)) in enumerate(zip(ωs, ws))
            A[j + 1, j + 1] = x ^ 2
            A[1, j + 1] = w
            A[j + 1, 1] = x
            B[j + 1, j + 1] = one(eltype(B))
        end
        
        iscomplex(x) = !isreal(x)
        ret = filter(iscomplex, map(sqrt ∘ complex, filter(isfinite, eigvals(A, B))))
        #ret = map(sqrt ∘ complex, filter(isfinite, eigvals(A, B)))
        append!(ret, -ret)
        sort!(ret, by = imag)
    end
    residues = [
        sum(
            w * (S / (pole - x) - S̄ / (pole + x))
            for (w, x, S, S̄) in zip(ws, ωs, Ss, S̄s)
        ) / sum(
            w * (-inv((pole - x) ^ 2) + inv((pole + x) ^ 2))
            for (w, x) in zip(ws, ωs)
        )
        for pole in poles
    ]
    constant = sum(w * (S - S̄) for (w, S, S̄) in zip(ws, Ss, S̄s)) + I * δ / 2
    return poles, residues, constant
    =#
end
