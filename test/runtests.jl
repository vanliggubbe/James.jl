using Jame
using Test
using Random
using LinearAlgebra


@testset "Bose factorization" begin
    for T in [1, 2, 3]
        f = bose_factor(T, 1e-8)
        g_exact(x) = (coth(x / 2 / T) + 1) * x / 2

        for x in LinRange(-10 * T, 10 * T, 8)
            @test isapprox(abs2(f(x)) / x, g_exact(x) / x; atol = 1e-4, rtol = 1e-5)
        end
    end 
end

#=
@testset "BSD factorization" begin
    J = CausalBSD([1 1], ones(1, 1), [1 1])
    F = FactorizedBSD(J)

    for x in LinRange(-10, 10, 8)
        @test isapprox(J(x), F(x))
    end
end
=#

@testset "BCF factorization" begin
    rng = MersenneTwister(1)
    # generate a random BSD
    M = randn(rng, 10, 10)
    M -= I * (real(first(eigvals(M))) * 1.1)
    J = FactorizedBSD(randn(rng, 10, 2), M, randn(rng, 10, 2))
    
    for T in [1.0, 2.0, 3.0]
        T = 1.0
        S_factor = bcf_factor(J, T, 1e-7)
        @test Jame.deep_eltype(S_factor.poly) <: Real
        @test eltype(S_factor.L) <: Real
        @test eltype(S_factor.M) <: Real
        for x in LinRange(-20, 20, 10)
            approx = S_factor(x)
            exact = J(x) * (coth(x / (2 * T)) + 1) / 2
            @test isapprox(approx * approx', exact; rtol = 1e-6, atol = 1e-6)
        end
    end
end

@testset "MarkovianEmbedding" begin
    rng = MersenneTwister(2)
    M = randn(rng, 10, 10)
    M -= I * (real(first(eigvals(M))) * 1.1)
    J = FactorizedBSD(randn(rng, 10, 2), M, randn(rng, 10, 2))

    T = 1.0
    me = MarkovianEmbedding(J, T, 1e-7)

    Ω = symplform(me)
    D_qq = kossakovski(me, :q, :q)
    D_cq = kossakovski(me, :c, :q)
    D_qc = kossakovski(me, :q, :c)
    D_qs = kossakovski(me, :q, :s)
    D_cc = kossakovski(me, :c, :c)
    D_cs = kossakovski(me, :c, :s)
    D_ss = kossakovski(me, :s, :s)
    
    H_qq = hamiltonian(me, :q, :q)
    H_qc = hamiltonian(me, :q, :c)
    H_qs = hamiltonian(me, :q, :s)
    H_cs = hamiltonian(me, :c, :s)
    H_ss = hamiltonian(me, :s, :s)

    M = [
        (-Ω * (H_qq + imag(D_qq)))  (-Ω * H_qc);
        (2 * real(D_cq))            drift(me)
    ]
    D = [
        (-Ω * real(D_qq) * Ω)   (-Ω * imag(D_qc));
        (-imag(D_cq) * Ω)       (D_cc)
    ]
    X = [
        (H_qs - imag(D_qs));
        H_cs
    ]
    P = [
        (-Ω * (H_qs + imag(D_qs)));
        2 * real(D_cs)
    ]
    Q = [
        (-Ω * real(D_qs));
        (-imag(D_cs))
    ]

    for ω in LinRange(-50, 50, 30)
        Σ_R = -im * X' * ((ω * I + im * M) \ P) + H_ss + imag(D_ss)
        Σ_K = (
            -im * X' * ((ω * I + im * M) \ (D * ((ω * I - im * M') \ X)))
            - X' * ((ω * I + im * M) \ Q)
            + Q' * ((ω * I - im * M') \ X)
            -im * real(D_ss)
        )

        @test isapprox((Σ_R - Σ_R') * 0.5im, J(ω); atol = 1e-6, rtol = 1e-6) 
        @test isapprox(Σ_K, -Σ_K')
        @test isapprox(
            Hermitian(im * Σ_K),
            J(ω) * coth(ω / (2 * T));
            atol = 1e-5, rtol = 1e-5
        )
    end
end
