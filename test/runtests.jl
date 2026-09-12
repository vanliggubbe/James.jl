using Jame
using Test
using Random
using LinearAlgebra

rng = MersenneTwister(0)

@testset "Bose approximation" begin
    for T in [1, 2]
        xs, ws, fs = Jame.aaa_bose(T)
        f_approx(z) = sum(
            f * w / (z - x) - f * w / (z + x)
            for (x, w, f) in zip(xs, ws, fs)
        ) / sum(
            w / (z - x) - w / (z + x)
            for (x, w) in zip(xs, ws)
        )
        f_exact(z) = (coth(z / (2 * T)) - 2 * T / z) / z
        for x in LinRange(-10 * T, 10 * T, 8)
            @test (f_approx(x) - f_exact(x)) < sqrt(eps(Float64))
        end
    end
end

@testset "Bose factorization" begin
    for T in [1, 2, 3]
        f = let (l, M, r, reg) = bose_factor(T; ε = 1e-8, δ = 5e-6, aaa_kwargs = (
                norm_weight = add_background(1e-3, lorenzian(20 * T)),
            ))
            x -> (l' * ((x * I + im * M) \ r) + reg[1] + reg[2] * x)
        end
        g_exact(x) = (coth(x / 2 / T) + 1) * x / 2

        for x in LinRange(-10 * T, 10 * T, 8)
            @test isapprox(abs2(f(x)) / x, g_exact(x) / x; atol = 1e-4, rtol = 1e-5)
        end
    end 
end

@testset "BSD factorization" begin
    J = CausalBSD([1 1], ones(1, 1), [1 1])
    F = FactorizedBSD(J)

    for x in LinRange(-10, 10, 8)
        @test isapprox(J(x), F(x))
    end
end

@testset "BCF factorization" begin
    # generate a random BSD
    M = randn(rng, 10, 10)
    M -= I * (real(first(eigvals(M))) * 1.1)
    J = FactorizedBSD(randn(rng, 10, 2), M, randn(rng, 10, 2))
    
    for T in [1.0, 2.0, 3.0]
        T = 1.0
        W, L, M, R = bcf_factor(J, T, 100.0; aaa_kwargs = (finite = true,)) 
        @test eltype(W) <: Real
        @test eltype(L) <: Real
        @test eltype(M) <: Real
        for x in LinRange(-20, 20, 10)
            approx = W + L' * ((x * I + im * M) \ R)
            exact = J(x) * (coth(x / (2 * T)) + 1) / 2
            @test isapprox(approx * approx', exact; rtol = 1e-6, atol = 1e-6)
        end
    end
end

@testset "MarkovianEmbedding" begin
    M = randn(rng, 10, 10)
    M -= I * (real(first(eigvals(M))) * 1.1)
    J = FactorizedBSD(randn(rng, 10, 2), M, randn(rng, 10, 2))

    T = 1.0
    me = MarkovianEmbedding(J, T, 100.0; aaa_kwargs = (finite = true,))

    Ω = symplform(me)
    D_qq = diffusion(me, :q, :q)
    D_cq = diffusion(me, :c, :q)
    D_qc = diffusion(me, :q, :c)
    D_qs = diffusion(me, :q, :s)
    D_cc = diffusion(me, :c, :c)
    D_cs = diffusion(me, :c, :s)
    D_ss = diffusion(me, :s, :s)
    
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
        @test isapprox(im * Σ_K, J(ω) * coth(ω / (2 * T)); atol = 1e-6, rtol = 1e-6)
    end
end
