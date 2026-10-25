using James
using LinearAlgebra
using KahanSummation

# AAA algorithm on a complex circle
function aaa_circle(
    f :: Function;
    tol = 1e-9,
    n_split :: Function = Base.Fix1(max, 3) ∘ Base.Fix1(-, 40),
    n_iter :: Int = 100,
)
    RTYPE = real(float(typeof(f(-1))))
    xs = collect(LinRange(RTYPE(0), 2 * RTYPE(π), n_split(0) + 2))[2 : end - 1]
    zs = expm1.(im * xs)
    fs = f.(zs)
    gs = 2 * fs
    js = collect(1 : length(xs))    # indices of the probe points
    αs = eltype(js)[]               # indices of the support points

    local ws
    inside(x) = isone(abs(x), Approx())
    for it in 1 : n_iter
        er, j = let
            fun(j) = abs(fs[j] - gs[j])
            findmax(fun, js)
        end
        if er < tol
            break
        end
        jj = js[j]
            
        # add new support points
        # find left and right points closest to the support point to be added
        x = xs[jj]
        yl, yr = let left = filter(<(x), xs[αs]), right = filter(>(x), xs[αs])
            maximum(left; init = RTYPE(0)), minimum(right; init = 2 * RTYPE(π))
        end
        push!(αs, jj)

        # delete all the probe points between supports
        cur = 0
        nleft = n_split(it) - 1
        nright = n_split(it) - 1
        for i in eachindex(js)
            if xs[js[i]] < yl || xs[js[i]] > yr
                cur += 1
                js[cur] = js[i]
            end
        end
        resize!(js, cur)

        # split the intervals, add more points
        nn = n_split(it)
        for new_xs in [
                collect(LinRange(yl, x, nleft + 3)[begin + 1 : end - 1]),
                collect(LinRange(x, yr, nright + 3)[begin + 1 : end - 1])
        ]
            new_zs = expm1.(im * new_xs)
            new_fs = map(f, new_zs)
            append!(js, length(xs) .+ (1 : nn))
            append!(xs, new_xs)
            append!(zs, new_zs)
            append!(fs, new_fs)
            append!(gs, zero(fs))
        end

        ws = let A = [(fs[α] - fs[j]) / (zs[j] - zs[α]) for j in js, α in αs]
            _, _, V = svd(A)
            V[:, end]
        end

        for j in js
            gs[j] = sum_kbn(
                [fs[α] * w / (zs[j] - zs[α]) for (w, α) in zip(ws, αs)];
            ) / sum_kbn(
                [w / (zs[j] - zs[α]) for (w, α) in zip(ws, αs)];
            )
        end
    end
    zs[αs], ws, fs[αs]
end

let
    L = [1.0, 1.0, 2.0, 2.0]
    R = [1.0, 1.0, 2.0, 2.0]
    M = [
        0.1 -5      0       0;
        5   0.1     0.05    0;
        0   0.05    0.3     17;
        0   0       -17     0.3
    ]
    J = FactorizedBSD(L, M, R)
    T = 0.6

    # construct BCF factorization
    S_james, _ = bcf_factor(J, T; δ = 1e-8)

    poles = let Ω = 7.0
        # map real axis onto a circle in the complex plane
        ω(z) = im * Ω * (z + 2) / z
        S(z) = -J(ω(z)) / expm1(-ω(z) / T)

        zs, ws, fs = aaa_circle(S; tol = 1e-8)
        ps = James.barycentric_poles(zs, ws)

        filter(x -> imag(x) < real(zero(x)), ω.(ps))
    end

    println("Jame poles:")
    display(-im * eigvals(S_james.M))
    println()
    println("FP poles:")
    display(poles)
end
