function barycentric_poles(xs, ws, c = zero(promote_type(eltype(xs), eltype(ws))))
    n = length(xs)

    A = zeros(promote_type(eltype(xs), eltype(ws)), n + 1, n + 1)
    A[2 : end, 1] .= map(sqrt ∘ abs, ws)
    A[1, 2 : end] .= A[2 : end, 1] .* phase_factor.(ws)
    A[1, 1] = c
    for (i, x) in enumerate(xs)
        A[i + 1, i + 1] = x
    end

    # calculate generalized eigenvalues
    # using generalized Schur decomposition
    B = Matrix(one(eltype(A)) * I, n + 1, n + 1)
    B[1, 1] = zero(eltype(B))
    f = schur(A, B)

    return filter((!) ∘ isnan, filter(isfinite, f.α ./ f.β))
end

function scalar_aaa(
    f :: Function,
    a :: Real,
    b :: Real,
    include_a :: Bool = true,
    include_b :: Bool = true;
    force_poles = nothing,
    tol = 1000 * eps(promote_type(typeof(a), typeof(b))),
    n_split :: Function = Base.Fix1(max, 3) ∘ Base.Fix1(-, 30) ∘ identity,
    n_iter :: Int = 100,
    weight :: Function = Returns(1),
    x_map :: Function = identity
)
    @argcheck ispos(tol)
    @argcheck ispos(n_iter)

    xs = collect(LinRange(x_map(a), x_map(b), n_split(0) + 2)) .|> inverse(x_map)
    fs = f.(xs)
    gs = 2 * fs
    fpoles = if isnothing(force_poles)
        eltype(fs)[]
    else
        force_poles
    end

    js = collect(2 : length(xs) - 1)    # indices of the probe points
    αs = eltype(js)[]                   # indices of the support points
    include_a && push!(αs, 1)
    include_b && push!(αs, length(xs))

    inside(x) = a < x < b

    local ws
    local ws_best
    for it in 1 : n_iter
        # check if 
        poles = it == 1 ? eltype(xs)[] : filter(inside, real(filter(isreal, barycentric_poles(xs[αs], ws))))
        jj = if !isempty(poles)
            # add the pole with the largest residue
            res = [
                -sum_kbn(fs[αs] .* ws ./ (p .- xs[αs])) /
                sum_kbn(ws ./ (p .- xs[αs]) .^ 2)
                for p in poles
            ]
            _, k = findmax(abs, res .* weight.(poles))
            push!(xs, poles[k])
            push!(fs, f(xs[end]))
            push!(js, length(xs))
            js[end]
        else
            if it != 1
                ws_best = copy(ws)
            end
            er, j = let
                fun(j) = abs(fs[j] - gs[j]) * abs(weight(xs[j]))
                findmax(fun, js)
            end
            #@debug "$(xs[js[j]]) $(er)"
            if er < tol
                break
            end
            js[j]
        end
            
        # add new support points
        # find left and right points closest to the support point to be added
        x = xs[jj]
        yl, yr = let left = filter(<(x), xs[αs]), right = filter(>(x), xs[αs])
            maximum(left; init = a), minimum(right; init = b)
        end
        push!(αs, jj)

        # delete all the probe points between supports
        cur = 0
        nleft = n_split(it) - 1  # count(j -> yl < xs[j] < x, js)
        nright = n_split(it) - 1 # count(j -> x < xs[j] < yr, js)
        for i in eachindex(js)
            if xs[js[i]] < yl || xs[js[i]] > yr
                cur += 1
                js[cur] = js[i]
            end
        end
        resize!(js, cur)

        # split the intervals, add more points
        nn = ensure(n_split(it), ispos)
        for new_xs in [
                collect(LinRange(x_map(yl), x_map(x), nleft + 3)[begin + 1 : end - 1]) .|> inverse(x_map),
                collect(LinRange(x_map(x), x_map(yr), nright + 3)[begin + 1 : end - 1]) .|> inverse(x_map)
        ]
            new_fs = map(f, new_xs)
            append!(js, length(xs) .+ (1 : nn))
            append!(xs, new_xs)
            append!(fs, new_fs)
            append!(gs, zero(fs))
        end


        C = [inv(xs[j] - xs[α]) for j in js, α in αs]
        ws = let A = [abs(weight(xs[j])) * (fs[j] - fs[α]) for j in js, α in αs] .* C
            Q = if length(fpoles) < length(αs)
                N = zeros(eltype(fs), length(αs), length(fpoles)) 
                for (i, p) in enumerate(fpoles)
                    if isfinite(p)
                        N[:, i] .= inv.(p .- xs[αs])
                    else
                        N[:, i] .= one(eltype(N))
                    end
                end
                Q, _ = qr(N)
                Q[:, length(fpoles) + 1 : end]
            else
                I
            end
            _, _, V = svd(A * Q)
            Q * V[:, end]
        end

        gs[js] .= xmul(C, (ws .* fs[αs])) ./ xmul(C, ws)
    end
    αs = αs[1 : length(ws_best)]

    xs[αs], ws_best, fs[αs]
end

# slow but allegedly accurate
function xmul(A :: Matrix, b :: Vector)
    c = Vector{promote_type(eltype(A), eltype(b))}(undef, size(A, 1))
    @inbounds for (j, a) in enumerate(eachrow(A))
        c[j] = sum_kbn([p * q for (p, q) in zip(a, b)])
    end
    return c
end
