module James

using LinearAlgebra
using Polynomials
using BlockArrays
using ArgCheck
using DescriptorSystems

include("utils.jl")
include("aaa.jl")
include("bsd.jl")
include("bose.jl")

export ConstFun, gaussian, lorenzian, box, add_background
export bose_factor
export BathSpectralDensity
export correlation_factorization
export HME_matrices
export bath_correlation_function

function correlation_factorization(J, T, Λ; ε = 1e-12)
    # factorize bose approximation
    p₊ = bose_factor(T, Λ; ε)
    @assert abs(imag(p₊.num.c)) < 1e-12
    W = real(p₊.num.c) * (J.L' * J.R)
    L = [J.L]
    M = [J.M]
    R = [p₊(-im * J.M) * J.R]
    for (pol, res) in residues(p₊)[2]
        Q, L′ = qr(J.L)
        R′ = first(res) * (Q') * ((pol * I + im * J.M) \ J.R)
        push!(L, L′[1 : size(L′, 2), :])
        push!(M, -imag(pol) * I(size(J.L, 2)))
        push!(R, R′[1 : size(L′, 2), :])
    end
    return (
        W,
        reduce(vcat, L),
        reduce(
            (x, y) -> [x zeros(size(x, 1), size(y, 2)); zeros(size(y, 1), size(x, 2)) y],
            M
        ),
        reduce(vcat, R)
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
