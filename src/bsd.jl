struct FactorizedBSD{
    T <: Real,
    ML <: AbstractMatrix{T},
    MM <: AbstractMatrix{T}, 
    MR <: AbstractMatrix{T}
}
    L :: ML
    M :: MM
    R :: MR

    function FactorizedBSD(L, M, R)
        @argcheck ishurwitz(M)
        new{eltype(L), typeof(L), typeof(M), typeof(R)}(L, M, R)
    end
end

spectral_factor(J :: FactorizedBSD, ω) = J.L' * ((ω * I + im * J.M) \ J.R)

_J_call(J :: FactorizedBSD, ω) = ω * spectral_factor(J, ω) * (spectral_factor(J, conj(ω)))'
(J :: FactorizedBSD)(ω) = _J_call(J, ω)
(J :: FactorizedBSD)(ω :: Real) = Hermitian(_J_call(J, ω))

"""
    FactorizedBSD(J :: Function, Ω :: Real = 1.0)

Construct a rational approximation of the bath spectral density and factorize it in form J(ω) = ω V(ω) [V(ω̄)]ᴴ. Approximation is construct using symmetrized AAA algorithm. 
"""
function FactorizedBSD(
    J :: Function,
    Ω :: Real = 1.0;
    ε :: Real = 1e-12,
    aaa_kwargs = (
        norm_weight = add_background(10 * ε, lorenzian(Ω)),
    ),
    rank_threshold = sqrt(eps(Float64))
)
    @argcheck ispos(rank_threshold)

    bsd(x :: Real) = ensure(J(x) / x, isposdef ∘ Hermitian, "Bath spectral density is not positive definite at positive frequencies")
    xs, ws, fs  = aaa_symm(bsd, Ω; ε = ε / 2, f_symm = conj, w_symm = [1 0; 0 -1], aaa_kwargs...)
    pol, res    = _bsd_pol_res(xs, ws, fs)
    P, A, Q     = _bsd_realization(pol, res; residue_sv_threshold = rank_threshold)
    Σ           = _bsd_riccati_solution(P, A, Q, rank_threshold)

    Λ, U = eigen(Hermitian(im * A * Σ - Σ * A' * im); sortby = (-))
    @assert Λ[end] > -rank_threshold
    Y = let rk = findlast(>(rank_threshold), Λ)
        U[:, 1 : rk] * Diagonal(sqrt.(Λ[1 : rk]))
    end

    return P, A, Y
end

# find pole-residue expansion
function _bsd_pol_res(xs, ws, fs)
    DTYPE = real(promote_type(eltype(xs), eltype(ws)))
    n = length(xs)

    A = zeros(DTYPE, 2 * n + 1, 2 * n + 1)
    @inbounds for (i, (x, w)) in enumerate(zip(xs, ws))
        A[2 * i + 0, 2 * i + 1] = x
        A[2 * i + 1, 2 * i + 0] = -x
        A[1, 2 * i + 0] = 2 * real(w)
        A[1, 2 * i + 1] = -2 * imag(w)
        A[2 * i + 0, 1] = one(eltype(A))
    end
    B = zeros(DTYPE, 2 * n + 1, 2 * n + 1)
    B .= I(size(B, 1))
    B[1, 1] = zero(eltype(B))
    pol = im * filter(isfinite, eigvals(A, B))

    res = [
        sum( 
            let wf = w * f
                (wf / (p - x)) + (conj(wf) / (p + x))
            end
            for (x, w, f) in zip(xs, ws, fs)
        ) / sum(
            -w / (p - x) ^ 2 - conj(w) / (p + x) ^ 2
            for (x, w) in zip(xs, ws)
        ) for p in pol
    ]
    indices = findall(isneg ∘ imag, pol)
    pol[indices], res[indices]
end

# construct a complex realization of a retarded part of bath spectral density
function _bsd_realization(poles, residues; residue_sv_threshold = sqrt(eps(real(eltype(poles), deep_eltype(residues)))))
    P = Matrix{deep_eltype(residues)}(undef, size(first(residues), 1), 0)
    Q = Matrix{deep_eltype(residues)}(undef, size(first(residues), 1), 0)
    A = eltype(poles)[]
    for (p, r) in zip(poles, residues)
        U, S, V = svd(r)
        rk = findlast(>(residue_sv_threshold), S)
        if !(rk isa Nothing)
            U = U[:, 1 : rk] * Diagonal(sqrt.(S[1 : rk]))
            V = V[:, 1 : rk] * Diagonal(sqrt.(S[1 : rk]))

            append!(A, fill(p, rk))
            P = hcat(P, U)
            Q = hcat(Q, V)
        end
    end
    Matrix(P'), Diagonal(A), Matrix(Q')
end

function _bsd_riccati_solution(P, A, Q, threshold :: Real)
    # solve Q = i Σ P for Σ
    H = let H = (im * P)' * Q
        Hermitian((H + H') / 2)
    end

    # check if H is positive semidefinite
    Λ, U = eigen(H; sortby = (-))
    @assert Λ[end] > -threshold

    # check whether ker(Q) is ker(H)
    rk = let 
        rk_H = findlast(>(threshold), Λ)
        (_, S, V) = svd(Q)
        rk_Q = findlast(>(threshold), S)
        @assert rk_H == rk_Q
        @assert rank([U[:, rk_H + 1 : end] V[:, rk_Q + 1 : end]]; atol = threshold) == size(H, 2) - rk_H
        rk_H
    end

    # Get solution in form Σ = W W⁺
    W = Q * (U[1 : rk, :])' * Diagonal(map(inv ∘ sqrt, Λ[1 : rk]))

    # Find Z such that i A (Σ + Z) - i (Σ + Z) A⁺ is positive semidefinite
    # Z must be represented as N Y N⁺ where N is basis of range complement of P
    # Σ + Z must also be positive semidefinite
    F = let F = im * A * Hermitian(W * W')
        Hermitian(F + F')
    end
    # range and it's complement for matrix P
    R, N = let (U, S, _) = svd(P)
        rk = findlast(>(threshold), S)
        U[:, begin : rk], nullspace(P')
    end
    
    # split F into block structure
    F_RR = Hermitian(R' * F * R)

    F_NN = Hermitian(N' * F * N)
    A_NN = N' * A * N

    F_RN = R' * F * N
    F_NR = F_RN'
    A_RN = R' * A * N

    # TODO add case of semidefinite F_RR with additional checks on kernel
    @assert isposdef(F_RR)
    # F_RR                  (F_RN + i A_RN Y)
    # (F_NR - i Y A'_NR)    (F_NN + i A_NN Y - i Y A'_NN)
    # need to check positive semidefiniteness of 
    # F_NN + (i A_NN) Y + Y (i A_NN)' - (F_NR - i Y A'_NR) F_RR⁻¹ (F_RN + i A_RN Y)
    # use Riccati equation solver
    (Y,) = arec((im * A_NN)', (im * A_RN)', zero(F_NN), F_RR, F_NN, F_NR; as = true)
    return Hermitian(W * W' + N * Y * N')
end
