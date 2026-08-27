using James
using Test

@testset "Bose approximation" begin
    for T in [1, 2]
        xs, ws, fs = James.aaa_bose(T)
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
        f = bose_factor(T; ε = 1e-8, δ = 5e-6, aaa_kwargs = (
            norm_weight = (x -> 1e-3 + (20 * T) ^ 2 / (x ^ 2 + (20 * T) ^ 2)),
        ))
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
