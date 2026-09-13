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

struct CausalBSD{
    T <: Real,
    S <: Union{T, Complex{T}},
    ML <: AbstractVecOrMat{T},
    MM <: AbstractMatrix{T},
    MR <: AbstractVecOrMat{S}
}
    L :: ML
    M :: MM
    R :: MR

    function CausalBSD(L, M, R, approx :: AbstractApprox = Approx(L, M, R))
        @argcheck ishurwitz(M, Val(false))
        @argcheck ishermitian(L' * (M \ R), approx)
        @argcheck ispossemidef(L' * R + R' * L, approx)

        T = promote_type(eltype(L), eltype(M), real(eltype(R)))
        S = promote_type(T, eltype(R))

        L_new = eltype(L) != T ? convert.(T, L) : L
        M_new = eltype(M) != T ? convert.(T, M) : M
        R_new = eltype(R) != S ? convert.(S, R) : R

        new{T, S, typeof(L_new), typeof(M_new), typeof(R_new)}(L_new, M_new, R_new)
    end
end

spectral_factor(J :: FactorizedBSD, ω) = J.L' * ((ω * I + im * J.M) \ J.R)

_J_call(J :: FactorizedBSD, ω) = ω * spectral_factor(J, ω) * (spectral_factor(J, conj(ω)))'
(J :: FactorizedBSD)(ω) = _J_call(J, ω)
(J :: FactorizedBSD)(ω :: Real) = Hermitian(_J_call(J, ω))

(J :: CausalBSD)(ω) = J.L' * ((ω * I + im * J.M) \ J.R) + J.R' * ((ω * I - im * J.M') \ J.L)
(J :: CausalBSD)(ω :: Real) = let X = J.L' * ((ω * I + im * J.M) \ J.R); Hermitian(X + X') end

function FactorizedBSD(
        J :: CausalBSD{T}, approx :: AbstractApprox = Approx(T)
) where {T}
    Σ = _bsd_riccati_solution(J.L, J.M, J.M \ J.R, approx)
    @check isapprox(J.M \ J.R, Σ * J.L, approx)

    # check positive semidefiniteness of (M Σ + Σ M⁺)
    # and get it's square root
    R = let (Λ, U) = eigen(Hermitian(J.M * Σ + Σ * J.M'); sortby = (-))
        atol = get(approx.kw, :atol, default_atol(Λ))
        rtol = get(approx.kw, :rtol, default_rtol(atol, Λ))

        @check Λ[end] > -max(atol, rtol * Λ[begin])
        rk = findlast(>(max(atol, rtol * Λ[begin])), Λ)
        R = U[:, 1 : rk] * Diagonal(sqrt.(Λ[1 : rk]))
        #@check size(J.L, 2) == length(Λ) || Λ[size(J.L, 2) + 1] < max(atol, Λ[begin] * rtol)

        # assume we've got an outer factor
        # make the right thing real-valued
        U, S, V = svd(R)

        Q = transpose(R) * U * Diagonal(map(inv, S)) * V'
        # matrix Q should be symmetric and unitary at the same time
        @check issymmetric(Q, approx)
        @check isunitary(Q, approx)

        S, = takagi(Symmetric(Q))
        R = R * S
        # product R * S should be real
        @check isreal(R, approx)
        real(R)
    end

    return FactorizedBSD(J.L, J.M, R)
end

"""
    CausalBSD(J :: Function, Ω :: Real = 1.0;)

Construct a rational approximation of the bath spectral density and express it in form

    J(ω) = Lᵀ (ω + i M)⁻¹ R + Rᴴ (ω - i Mᵀ)⁻¹ L

where `L` and `M` are real-valued matrices, and matrix `R` may be complex-valued. The approximation is construct using continuouv symmetrized version of AAA algorithm. 
"""
function CausalBSD(
    J :: Function,
    Ω :: Real = 1.0;
    atol :: Real = default_atol(Ω),
    rtol :: Real = default_rtol(atol, Ω),
    aaa_kwargs = (
        norm_weight = add_background(10 * atol, lorenzian(Ω)),
    ),
)
    @argcheck tol_check(atol, rtol)

    # make AAA approximation
    xs, ws, fs  = let
        validate(J) = ispossemidef(J; atol, rtol)
        bsd(x :: Real) = ensure(J(x) / x, validate, "Bath spectral density is not positive definite at positive frequencies")
        aaa_symm(bsd, Ω; atol = atol / 2, rtol, f_symm = conj, w_symm = [1 0; 0 -1], aaa_kwargs...)
    end
    pol, res    = _bsd_pol_res(xs, ws, fs)
    P, M, Q     = _bsd_realization(pol, res; atol, rtol)

    #L, A, R, reg = _bsd_gen_schur(xs, ws)
    #return L, A, R, reg

    return P, M, Q
    display(P' * Q)
    display(P' * im * M * Q)
    return CausalBSD(P, M, -im * M * Q)
end

# that's for future
# try to avoid eigen decomposition
# and work in real numbers
function _bsd_gen_schur(xs, ws)
    DTYPE = real(promote_type(eltype(xs), eltype(ws)))
    n = length(xs)

    A = zeros(DTYPE, 2 * n + 1, 2 * n + 1)
    @inbounds for (i, (x, w)) in enumerate(zip(xs, ws))
        A[2 * i + 0, 2 * i + 1] = -x
        A[2 * i + 1, 2 * i + 0] = x
        A[1, 2 * i + 0] = 2 * real(w)
        A[1, 2 * i + 1] = -2 * imag(w)
        A[2 * i + 0, 1] = one(eltype(A))
    end
    B = zeros(DTYPE, 2 * n + 1, 2 * n + 1)
    B .= I(size(B, 1))
    B[1, 1] = zero(eltype(B))
    

    k, L, A, E, R = let F = schur(A, B), x = [one(DTYPE); zeros(DTYPE, 2 * n)]
        k = findlast(iszero, diag(F.T))
        if isodd(2 * n + 1 - k)
            k += 1
        end
        (
            k,
            BlockedArray(F.right' * x, [k, 2 * n + 1 - k]),
            BlockedArray(F.S, [k, 2 * n + 1 - k], [k, 2 * n + 1 - k]),
            BlockedArray(F.T, [k, 2 * n + 1 - k], [k, 2 * n + 1 - k]),
            BlockedArray(F.left' * x, [k, 2 * n + 1 - k])
        )
    end
    #return L, A, E, R, k

    inv_11 = let X = [inv(A[Block(1), Block(1)])];
        append!(X, fill(E[Block(1), Block(1)] / A[Block(1), Block(1)], k))
        Y = cumprod(X)
        Y[begin : findlast((!) ∘ iszero, Y)]
    end
    poly_12 = let a = inv_11, b = [-A[Block(1), Block(2)], E[Block(1), Block(2)]]
        c = Matrix{DTYPE}[]
        for (i, aa) in enumerate(a)
            for (j, bb) in enumerate(b)
                while i + j - 1 > length(c)
                    push!(c, zeros(DTYPE, size(aa, 1), size(bb, 2)))
                end
                mul!(c[i + j - 1], aa, bb, one(DTYPE), one(DTYPE))
            end
        end
        c
    end
    l = L[Block(2)]
    r = R[Block(2)]
    reg = [L[Block(1)]' * p * R[Block(1)] for p in inv_11]

    X = E[Block(2), Block(2)] \ A[Block(2), Block(2)]
    cur_reg = DTYPE[]
    cur_sin = Matrix(one(DTYPE)I, size(X))

    for p in poly_12
        l += cur_sin' * p' * L[Block(1)]
        @inbounds for (j, r) in enumerate(cur_reg)
            reg[j] += L[Block(1)]' * p * r * R[Block(2)]
        end
        # recalculate
        cur_reg = [[zero(X)]; cur_reg]
        cur_reg[1] -= cur_sin / E[Block(2), Block(2)]
        cur_sin *= X
    end

    U, S, V = svd(E[Block(2), Block(2)])
    #sqrt_S = Diagonal(map(sqrt, S))
    isqrt_S = Diagonal(map(sqrt ∘ inv, S))
    a = isqrt_S * U' * A[Block(2), Block(2)] * V * isqrt_S
    r = isqrt_S * U' * r
    l = isqrt_S * V' * l
    return l, a, r, reg
    return l, A[Block(2), Block(2)], E[Block(2), Block(2)], r, reg
end

# find pole-residue expansion
function _bsd_pol_res(xs, ws, fs)
    DTYPE = real(promote_type(eltype(xs), eltype(ws)))
    n = length(xs)

    A = zeros(DTYPE, 2 * n + 1, 2 * n + 1)
    @inbounds for (i, (x, w)) in enumerate(zip(xs, ws))
        A[2 * i + 0, 2 * i + 1] = -x
        A[2 * i + 1, 2 * i + 0] = x
        A[1, 2 * i + 0] = 2 * real(w)
        A[1, 2 * i + 1] = -2 * imag(w)
        A[2 * i + 0, 1] = one(eltype(A))
    end
    B = zeros(DTYPE, 2 * n + 1, 2 * n + 1)
    B .= I(size(B, 1))
    B[1, 1] = zero(eltype(B))
    pol = filter(ispos ∘ real, filter(isfinite, eigvals(A, B; sortby = (-) ∘ real)))

    res = [
        sum( 
            let wf = w * f
                (wf / (-im * p - x)) + (conj(wf) / (-im * p + x))
            end
            for (x, w, f) in zip(xs, ws, fs)
        ) / sum(
            -w / (-im * p - x) ^ 2 - conj(w) / (-im * p + x) ^ 2
            for (x, w) in zip(xs, ws)
        ) for p in pol
    ]
    pol, res
end

# construct a complex realization of a retarded part of bath spectral density
function _bsd_realization(
        poles,
        residues,
        approx :: AbstractApprox = Approx(poles, residues)
)
    atol = get(approx.kw, :atol, default_atol(poles, residues))
    rtol = get(approx.kw, :rtol, default_rtol(atol, poles, residues))

    PTYPE = real(deep_eltype(poles))
    P = Matrix{deep_eltype(residues)}(undef, size(first(residues), 1), 0)
    Q = Matrix{deep_eltype(residues)}(undef, size(first(residues), 1), 0)
    A = eltype(poles)[]
    mask = zeros(Bool, length(poles))
    for (i, (p, r)) in enumerate(zip(poles, residues))
        # factorize the residue
        U, S, V = svd(r)
        rk = findlast(
            >(
                max(one(PTYPE), abs(real(p))) * 
                max(atol, rtol * S[begin])
            ), S
        )

        if !(rk isa Nothing)
            if isreal(p, approx)
                sqrt_S = Diagonal(map(sqrt, @view S[1 : rk]))
                U = U[:, 1 : rk] * sqrt_S
                V = V[:, 1 : rk] * sqrt_S
                @inbounds for j in axes(U, 2)
                    _, k = findmax(abs, @view U[:, j])
                    factor = unsafe_phase_factor(U[k, j])
                    @views U[:, j] *= conj(factor)
                    @views V[:, j] *= factor
                end

                append!(A, fill(complex(real(p)), rk))
                P = hcat(P, U)
                Q = hcat(Q, V)
            elseif !mask[i]
                j = let fun(x) = isapprox(x, p', approx)
                    findfirst(fun, poles)
                end
                mask[i] = true
                mask[j] = true
                # TODO
            end
        end
    end

    #=
    # transform matrix to the real valued
    T = zeros(eltype(A), length(A), length(A))
    for (i, a) in enumerate(A)
        if isreal(a)
            T[i, i] = one(eltype(T))
        elseif !mask[i]
            # find conjugated pairs
            i1 = findall(==(a), A)
            i2 = findall(≈(a'), A)
            @assert length(i1) == length(i2)
            @assert allunique([i1; i2])
            mask[i1] .= true
            mask[i2] .= true
            T[i1, i1] .= I(length(i1)) 
            T[i2, i1] .= I(length(i1)) 
            T[i1, i2] .= -im * I(length(i1)) 
            T[i2, i2] .= im * I(length(i1)) 
        end
    end
    (P * T)', real((T \ Diagonal(A)) * T), T \ Q'
    =#
end

# find positive semidefinite Σ such that
# Q = Σ P
# A Σ + Σ A⁺ is positive semidefinite
# reduced to riccati solver
function _bsd_riccati_solution(
        P, A, Q,
        approx :: AbstractApprox = Approx(P, A, Q) 
)
    if isempty(A)
        return Hermitian(copy(A))
    end
    # solve Q = Σ P for Σ
    H = let H = P' * Q
        @check ishermitian(H, approx)
        Hermitian((H + H') / 2)
    end

    # check if H is positive semidefinite
    atol = get(approx.kw, :atol, default_atol(P, A, Q))
    rtol = get(approx.kw, :rtol, default_rtol(atol, P, A, Q))
    Λ, U = eigen(H; sortby = (-))
    @check Λ[end] > -max(atol, Λ[begin] * rtol)

    # check whether ker(Q) is ker(H)
    rk = let 
        no_fun(x) = x
        no_fun(:: Nothing) = 0

        rk_H = findlast(>(max(atol, rtol * Λ[1])), Λ) |> no_fun
        (_, S, V) = svd(Q)
        rk_Q = findlast(>(max(atol, rtol * S[1])), S) |> no_fun

        @check rk_H == rk_Q
        @check rank(
            [U[:, rk_H + 1 : end] V[:, rk_Q + 1 : end]];
            atol, rtol
        ) == size(H, 2) - rk_H

        rk_H
    end

    # Get solution in form Σ = W W⁺
    W = Q * U[:, 1 : rk] * Diagonal(map(inv ∘ sqrt, Λ[1 : rk]))

    # Find Z such that A (Σ + Z) + (Σ + Z) A⁺ is positive semidefinite
    # Z must be represented as N Y N⁺ where N is basis of range complement of P
    # Σ + Z must also be positive semidefinite
    F = let F = A * Hermitian(W * W')
        Hermitian(F + F')
    end
    # range and it's complement for matrix P
    N = nullspace(P'; atol, rtol)
    R = nullspace(N'; atol, rtol)
    
    # split F into block structure
    R_F_R = Hermitian(R' * F * R)

    N_F_N = Hermitian(N' * F * N)
    N_A_N = N' * A * N

    R_F_N = R' * F * N
    N_F_R = R_F_N'
    R_A_N = R' * A * N

    # get null space of F block which corresponds to the range of P
    RN = nullspace(R_F_R; atol, rtol) 
    RR = nullspace(RN'; atol, rtol)

    if !isempty(RR)
        RR_F_RR = Hermitian(RR' * R_F_R * RR)
        # strict posdef since we work in the range subspace
        @check isposdef(RR_F_RR)

        N_F_RR = N_F_R * RR
        RR_A_N = RR' * R_A_N
        #RN_A_N = RN' * R_A_N

        # R_F_R                  (R_F_N + R_A_N Y)
        # (N_F_R + Y N_A'_R)     (N_F_N + N_A_N Y + Y N_A'_N)
        # need to check positive semidefiniteness of 
        # N_F_N + N_A_N Y + Y N_A_N' - (N_F_R + Y R_A_N') R_F_R⁻¹ (R_F_N + R_A_N Y)
       
        # use Riccati equation solver
        (Y,) = arec(N_A_N', RR_A_N', zero(N_F_N), RR_F_RR, N_F_N, N_F_RR; as = true)
        if !isempty(RN)
            error("Not implemented :(")
        end
        return Hermitian(W * W' + N * Y * N')
    else
        Y = _bsd_riccati_solution(R_A_N', N_A_N, -N_F_R, approx)
        return Hermitian(W * W' + N * Y * N')
    end
end
