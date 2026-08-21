using James
using Test

@testset "bose approximation" begin
    for T in [1, 2]
        xs, ws, fs = James.aaa_bose(T, sqrt(eps(Float64)))
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
        #=
        f_approx = bose_factor(T; δ = 1e-4)
        g_exact(x) = (coth(x / (2.0 * T)) * x + x) / 2
        for x in LinRange(-5 * T, 5 * T, 4)
            println(x, " ", abs2(f_approx(x)), " ", g_exact(x))
        end
        =#
    end
end
