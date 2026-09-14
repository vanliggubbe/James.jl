struct MarkovianEmbedding{
    RTYPE <: Real,
    ITYPE,
    MK <: Hermitian{<: Union{RTYPE, Complex{RTYPE}}},
    MH <: Symmetric{RTYPE},
    MM <: AbstractMatrix{RTYPE}
}

    index_s :: ITYPE
    index_q :: ITYPE
    index_c :: ITYPE

    koss :: MK
    hmlt :: MH 
    drft :: MM
end

_index(me :: MarkovianEmbedding, i :: AbstractString) = (
    i in ("s", "sys", "system") ? me.index_s : (
        i in ("q", "quant", "quantum") ? me.index_q : (
            i in ("c", "cl", "class", "classical") ? me.index_c :
            throw(ArgumentError("Unknown block index: $i"))
        )
    )
)

_index(me :: MarkovianEmbedding, i :: Symbol) = (
    i in (:s, :sys, :system) ? me.index_s : (
        i in (:q, :quant, :quantum) ? me.index_q : (
            i in (:c, :cl, :class, :classical) ? me.index_c :
            throw(ArgumentError("Unknown block index: $i"))
        )
    )
)
"""
    kossakovski(me :: MarkovianEmbedding[, i, j])

Returns Kossakovski matrix of Markovian embedding `me`. If indices `i` and `j` are specified, returns corresponding block of the matrix. Possible values of `i` and `j` are:
- `"s"`, `"sys"`, `"system"`, `:s`, `:sys`, `:system` for the coupling operator block
- `"q"`, `"quant"`, `"quantum"`, `:q`, `:quant`, `:quantum` for quantum degrees of freedom of the embedding
- `"c"`, `"cl"`, `"class"`, `"classical"`, `:c`, `:cl`, `:class`, `:classical` for classical degrees of freedom of the embedding
"""
kossakovski(me :: MarkovianEmbedding) = me.koss
kossakovski(me :: MarkovianEmbedding, i, j) = me.koss[_index(me, i), _index(me, j)]

"""
    hamiltonian(me :: MarkovianEmbedding[, i, j])

Returns Hamiltonian matrix of Markovian embedding `me`. If indices `i` and `j` are specified, returns corresponding block of the matrix. Possible values of `i` and `j` are:
- `"s"`, `"sys"`, `"system"`, `:s`, `:sys`, `:system` for the coupling operator block
- `"q"`, `"quant"`, `"quantum"`, `:q`, `:quant`, `:quantum` for quantum degrees of freedom of the embedding
- `"c"`, `"cl"`, `"class"`, `"classical"`, `:c`, `:cl`, `:class`, `:classical` for classical degrees of freedom of the embedding
"""
hamiltonian(me :: MarkovianEmbedding) = me.hmlt
hamiltonian(me :: MarkovianEmbedding, i, j) = me.hmlt[_index(me, i), _index(me, j)]

"""
    drift(me :: MarkovianEmbedding)

Returns drift matrix of classical degrees of freedom of Markovian embedding `me`.
"""
drift(me :: MarkovianEmbedding) = me.drft

"""
    ndof(me :: MarkovianEmbedding[, i])

Returns size of Hamiltonian and Kossakovski matrices of Markovian embedding `me`. If index `i` is specified, returns size of the respective block. Possible values of `i` are
- `"s"`, `"sys"`, `"system"`, `:s`, `:sys`, `:system` for the coupling operator block
- `"q"`, `"quant"`, `"quantum"`, `:q`, `:quant`, `:quantum` for quantum degrees of freedom of the embedding
- `"c"`, `"cl"`, `"class"`, `"classical"`, `:c`, `:cl`, `:class`, `:classical` for classical degrees of freedom of the embedding
"""
ndof(me :: MarkovianEmbedding) = length(me.index_s) + length(me.index_q) + length(me.index_c)
ndof(me :: MarkovianEmbedding, i) = length(_index(me, i))

"""
    symplform(me :: MarkovianEmbedding)

Returns symplectic `Ω` form which specifies commutator matrix of the quantum degrees of freedom of Markovian embedding `me`:
    [x̂ⱼ, x̂ₖ] = i Ωⱼₖ
"""
symplform(me :: MarkovianEmbedding{T}) where {T} = kron(
    I(length(me.index_q) ÷ 2), 
    [zero(T) one(T); -one(T) zero(T)]
)

function bcf_factor(J :: FactorizedBSD, f_bose :: RationalPencil)
    # factorize bose approximation
    @argcheck isone(f_bose.c * im)
    @argcheck eltype(f_bose.L) <: Real
    @argcheck eltype(f_bose.M) <: Real

    d = size(J.R, 2)
    L = kron(f_bose.L, I(d))
    M = kron(f_bose.M, I(d))
    R = kron(f_bose.R, I(d))
    reg = f_bose.poly

    # in principle should always be valid, but who knows
    @argcheck isreal(reg[2], Approx(reg[2]))

    LTYPE = eltype(J.L)
    MTYPE = promote_type(eltype(J.M), eltype(M), eltype(J.R), eltype(L))
    return RationalPencil(
        [J.L' * J.R * real(reg[2]), ], -im,
        [
            J.L;
            zeros(LTYPE, size(L, 1), size(J.L, 2))
        ],
        [
            J.M                                     (-J.R * L');
            zeros(MTYPE, size(M, 1), size(J.M, 2))  M
        ],
        [
            (J.R * reg[1] - im * J.M * J.R * reg[2]);
            -im * R
        ]
    )
end

bcf_factor(J :: FactorizedBSD, T :: Real, args...; kwargs...) = bcf_factor(J, bose_factor(T, args...; kwargs...))


MarkovianEmbedding(J :: FactorizedBSD, T :: Real, args...; kwargs...) = MarkovianEmbedding(
    bcf_factor(J, T, args...; kwargs...)
)

function MarkovianEmbedding(
    S_factor :: RationalPencil,
    approx :: AbstractApprox = Approx(S_factor)
)
    @argcheck deep_eltype(first(S_factor.poly)) <: Real
    @argcheck deep_eltype(S_factor.L) <: Real
    @argcheck deep_eltype(S_factor.M) <: Real
    @argcheck isone(S_factor.c * im) 

    atol = get(approx.kw, :atol, default_atol(S_factor))
    rtol = get(approx.kw, :rtol, default_rtol(atol, S_factor))
    norm = get(approx.kw, :norm, frnorm)

    W = first(S_factor.poly)
    L = S_factor.L
    M = S_factor.M
    R = S_factor.R

    # solve Lyapunov equation for the presymplectic form
    # and reduce it to the Darboux form
    Θ = lyapc(M, -2 * imag(R * R'))
    rk, U, U_t, U_inv, U_tinv = let (U, S, V) = svd(Θ);
        ε = max(rtol * norm(Θ), atol)
        sup = findall(>(ε), S)

        @inbounds for i in sup
            if isodd(i)
                @views V[:, i] .= U[:, i + 1]
            end
        end
        s1 = Diagonal([s > ε ? sqrt(s) : one(s) for s in S])
        s2 = inv(s1)
        (length(sup), V * s1, s1 * V', s2 * V', V * s2)
    end

    # transform everything to the Darboux basis
    Θ = nullify!(U_inv * Θ * U_tinv; atol, rtol, norm)
    M = nullify!(U_inv * M * U; atol, rtol, norm)
    R = nullify!(U_inv * R; atol, rtol, norm)
    L = nullify!(U_t * L; atol, rtol, norm)

    # symplectic block
    Ω = (@view Θ[1 : rk, 1 : rk])
    ker = size(Θ, 1) - rk 

    # kossakovski matrix
    D = let T = [(-im * inv(Ω)) zeros(rk, ker); zeros(ker, rk) I(ker)], W = W, R = R
        X = [sqrt(2) * W; T * R];
        Hermitian(X * X')
    end

    # indices of quantum and classical noises
    i_q = 1 : rk
    i_c = rk .+ (1 : ker)

    # hamiltonian matrix
    K = @views (L[i_q, :] - Ω \ (2 * real(R) * W' - Θ * L)[i_q, :]) / sqrt(2)
    H = Symmetric([
        zeros(size(W))          K'                      (sqrt(2) * L[i_c, :]');
        K                       -symm(Ω \ M[i_q, i_q])  (-Ω \ M[i_q, i_c]);
        (sqrt(2) * L[i_c, :])   (-Ω \ M[i_q, i_c])'     zeros(ker, ker)
    ])
    return MarkovianEmbedding(
        1 : size(W, 1),
        size(W, 1) .+ (1 : rk),
        (size(W, 1) + rk) .+ (1 : ker),
        D, H, M[rk + 1 : end, rk + 1 : end]
    )
    #=
    return (
        Matrix(Ω), 
        BlockedArray(D, [size(L, 2), rk, ker], [size(L, 2), rk, ker]),
        BlockedArray(H, [size(L, 2), rk, ker], [size(L, 2), rk, ker]),
        M[rk + 1 : end, rk + 1 : end]
    )
    =#
end

