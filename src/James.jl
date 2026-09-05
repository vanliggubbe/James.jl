module James

using LinearAlgebra
using Polynomials
using BlockArrays
using ArgCheck
using MatrixEquations
import KahanSummation: sum_kbn

include("utils.jl")
include("aaa.jl")
include("bsd.jl")
include("bose.jl")

export ConstFun, gaussian, lorenzian, box, add_background
export bose_factor
export CausalBSD, FactorizedBSD
export correlation_factorization
export HME_matrices

function correlation_factorization(J :: FactorizedBSD, T :: Real; kwargs...)
    # factorize bose approximation
    L, M, R, reg = let (L, M, R, reg) = bose_factor(T; kwargs...), d = size(J.R, 2)
        kron(L, I(d)), kron(M, I(d)), kron(R, I(d)), reg
    end

    LTYPE = eltype(J.L)
    MTYPE = promote_type(eltype(J.M), eltype(M), eltype(J.R), eltype(L))
    return (
        J.L' * J.R * reg[2],
        [
            J.L;
            zeros(LTYPE, size(L, 1), size(J.L, 2))
        ],
        [
            J.M                                     (J.R * L');
            zeros(MTYPE, size(M, 1), size(J.M, 2))  M
        ],
        [
            (J.R * reg[1] - im * J.M * J.R);
            (-R)
        ]
    )
end

symm(a) = (a + transpose(a)) / 2
skew(a) = (a - transpose(a)) / 2


function HME_matrices(J, T, Λ; ε = 1e-12)
    W, L, M, R = correlation_factorization(J, T, Λ; ε)

    # solve Lyapunov equation for the presymplectic form
    # and reduce it to the Darboux form
    Θ = lyap(M, -2 * imag(R * R'))
    rk, U, U_t, U_inv, U_tinv = let (_, S, V) = svd(Θ);
        s1 = Diagonal([s > ε ? sqrt(s) : one(s) for s in S])
        s2 = inv(s1)
        (sum(S .> ε), V * s1, s1 * V', s2 * V', V * s2)
    end

    # transform everything to the Darboux basis
    almost_zero(x) = abs(x) < ε ? zero(x) : x
    Θ = map(almost_zero, U_inv * Θ * U_tinv)
    M = map(almost_zero, U_inv * M * U)
    R = map(almost_zero, U_inv * R)
    L = map(almost_zero, U_t * L)

    # symplectic block
    Ω = (@view Θ[1 : rk, 1 : rk])
    ker = size(Θ, 1) - rk 


    # diffusion matrix
    D = let T = [(-im * inv(Ω)) zeros(rk, ker); zeros(ker, rk) I(ker)], W = W, R = R
        X = [sqrt(2) * W; T * R];
        Hermitian(X * X')
    end

    # indices of quantum and classical noises
    i_q = 1 : rk
    i_c = rk .+ (1 : ker)

    # hamiltonian matrix
    K = @views (L[i_q, :] - Ω \ (2 * real(R) * W' - Θ * L)[i_q, :]) / sqrt(2)
    H = [
        zeros(size(W))          K'                      (sqrt(2) * L[i_c, :]');
        K                       -symm(Ω \ M[i_q, i_q])  (-Ω \ M[i_q, i_c]);
        (sqrt(2) * L[i_c, :])   (-Ω \ M[i_q, i_c])'     zeros(ker, ker)
    ]
    return (
        Matrix(Ω), 
        BlockArray(D, [size(L, 2), rk, ker], [size(L, 2), rk, ker]),
        BlockArray(H, [size(L, 2), rk, ker], [size(L, 2), rk, ker]),
        M[rk + 1 : end, rk + 1 : end]
    )
end

end
