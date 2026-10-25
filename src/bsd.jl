struct FactorizedBSD{
    T <: Real,
    ML <: AbstractVecOrMat{T},
    MM <: AbstractMatrix{T}, 
    MR <: AbstractVecOrMat{T}
}
    L :: ML
    M :: MM
    R :: MR

    function FactorizedBSD(L, M, R)
        @argcheck ishurwitz(M, Val(false))

        T = promote_type(eltype(L), eltype(M), eltype(R))

        L_new = eltype(L) != T ? convert.(T, L) : L
        M_new = eltype(M) != T ? convert.(T, M) : M
        R_new = eltype(R) != T ? convert.(T, R) : R

        new{T, typeof(L_new), typeof(M_new), typeof(R_new)}(L_new, M_new, R_new)
    end
end

Base.eltype(:: Type{FactorizedBSD{T, ML, MM, MR}}) where {T, ML, MM, MR} = T
Base.eltype(:: FactorizedBSD{T}) where {T} = T

spectral_factor(J :: FactorizedBSD, ω) = J.L' * ((ω * I + im * J.M) \ J.R)
_J_call(J :: FactorizedBSD, ω) = ω * spectral_factor(J, ω) * (spectral_factor(J, conj(ω)))'

(J :: FactorizedBSD)(ω) = _J_call(J, ω)

_hermitian(x :: AbstractMatrix) = Hermitian(x)
_hermitian(x :: Number) = real(x)

function (J :: FactorizedBSD)(ω :: Real)
    f = spectral_factor(J, ω)
    return _hermitian(ω * f * f')
end

"""
    __norm_estimate(L, M, R)

Gives an estimate for 

    sup_`ω` ||`L`ᵀ (`ω` + i `M`)⁻¹ `R`||

Here ||⋅|| is a spectral matrix norm. `M` is assumed to be Hurwitz-unstable
"""
_norm_estimate(L, M, R) = min(
    __norm_estimate(L, M, R),
    __norm_estimate(conj(R), transpose(M), conj(L))
)

_opnorm(a :: AbstractMatrix) = opnorm(a)
_opnorm(a :: AbstractVector) = frnorm(a)
_opnorm(a :: Number) = abs(a)

function __norm_estimate(L, M, R)
    ret = real(promote_type(
        eltype(L),
        eltype(M),
        eltype(R)
    ))[]
    for A in (
            I(size(M, 1)),
            sqrt(M * M'), sqrt(M' * M),
            M * M', M' * M
    )
        P = lyapc(M', -2 * A)
        Λ, Ψ = eigen(Hermitian(P))
        Q = Diagonal(map(sqrt, Λ))
        isqP = Ψ * (Q \ Ψ')
        γ = inv(minimum(eigvals(isqP * A * isqP)))
        push!(ret, γ * _opnorm(isqP * L) * _opnorm(isqP \ R))
    end
    return minimum(ret)
end

function _norm_estimate_2(L, M, R)
    ret = real(promote_type(
        eltype(L),
        eltype(M),
        eltype(R)
    ))[]
    for A in (
            I(size(M, 1)),
            sqrt(M * M'), sqrt(M' * M),
            M * M', M' * M
    )
        P = lyapc(M', -2 * A)
        Λ, Ψ = eigen(Hermitian(P))
        Q = Diagonal(map(sqrt, Λ))
        isqP = Ψ * (Q \ Ψ')
        γ = inv(minimum(eigvals(isqP * A * isqP))) * _opnorm(isqP \ (M * isqP))
        push!(ret, γ * _opnorm(isqP * L) * _opnorm(isqP \ R))
    end
    return minimum(ret)

end

function _norm_estimate_freq(L, M, R)
    # ω Lᵀ (ω + i M)⁻¹ R = 
    # Lᵀ R - i Lᵀ (ω + i M)⁻¹ M R
    return _opnorm(L' * R) + min(
        _norm_estimate(L, M, M * R),
        _norm_estimate(M' * L, M, R),
        _norm_estimate_2(L, M, R)
    )
end

function _norm_estimate(L, M, C, R)
    # Lᵀ (ω + i M)⁻¹ C (ω - i Mᵀ)⁻¹ R
    X = lyapc(M, -C)
    return _norm_estimate(L, M, X * R) + _norm_estimate(X' * L, M', R)
end

function norm_estimate(J :: FactorizedBSD)
    L, M, R = J.L, J.M, J.R

    # norm w/o frequency factor
    A = min(
        _norm_estimate(L, M, R) ^ 2,
        _norm_estimate(L, M, R * R', L),
        _norm_estimate(R, M', L * L', R)
    )

    # norm with freq factor
    B = min(
        _norm_estimate_freq(L, M, R),
        _norm_estimate_freq(R, M', L)
    ) ^ 2

    B = min(
        B, _opnorm(L' * R) ^ 2 +
        2 * min(
            _norm_estimate(L, M, M * R * R' * L),
            _norm_estimate(M' * L, M, R * R' * L),
            _norm_estimate_2(L, M, R * R' * L)
        ) + min(
            _norm_estimate(L, M, M * R * R' * M', L),
            _norm_estimate(M' * L, M, R * R', M' * L),
            _norm_estimate(L, M, M * R * R', M' * L),
            _norm_estimate(M' * L, M, R * R' * M', L)
        )
    )

    B = min(
        B, _opnorm(R' * L) ^ 2 +
        2 * min(
            _norm_estimate(R, M', M' * L * L' * R),
            _norm_estimate(M * R, M', L * L' * R),
            _norm_estimate_2(R, M', L * L' * R)
        ) + min(
            _norm_estimate(R, M', M' * L * L' * M, R),
            _norm_estimate(M * R, M', L * L', M * R),
            _norm_estimate(L, M', M' * L * L', M * R),
            _norm_estimate(M * R, M', L * L' * M, R)
        )
    )
    return A, B
end
