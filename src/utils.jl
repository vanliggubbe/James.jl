# Auxiliary functions

const RealOrComplex{T} = Union{T, Complex{T}} where {T}
const AbstractScalarOrMatrix{T} = Union{T, AbstractMatrix{T}} where {T}

symm(a) = (a + transpose(a)) / 2 
skew(a) = (a - transpose(a)) / 2

deep_eltype(:: Type{T}) where {T} = let ET = eltype(T);
    (T == ET) ? T : deep_eltype(ET)
end

deep_eltype(:: T) where {T} = deep_eltype(T)

function ensure(x, f)
    @argcheck f(x)
    x
end

function ensure(x, f, msg)
    @argcheck f(x) msg
    x
end

#=
# checks
=#
@inline ispos(x :: Real) = (x > zero(x))
@inline isneg(x :: Real) = (x < zero(x))
@inline isnpos(x :: Real) = !ispos(x)
@inline isnneg(x :: Real) = !isneg(x)

default_atol(x...) = zero(float(real(promote_type(deep_eltype.(x)...))))
default_rtol(atol, x...) = (
    iszero(atol) ? 
    sqrt(eps(float(real(promote_type(deep_eltype.(x)...))))) :
    default_atol(x...)
)

@inline tol_check(a :: Real, r :: Real) = isnneg(a) && isnneg(r) && (ispos(a) || ispos(r))
@inline norm_check(A, n) = (iszero(A) == iszero(n)) && isnneg(n)

# interface to IsApprox.jl
function Approx(
    x, y...;
    atol :: Real = default_atol(x, y...),
    rtol :: Real = default_rtol(atol, x, y...),
    norm :: Function = frnorm
) 
    @argcheck tol_check(atol, rtol)
    return Approx(; atol, rtol, norm)
end

function EachApprox(
    x, y...;
    atol :: Real = default_atol(x, y...),
    rtol :: Real = default_rtol(atol, x, y...)
)
    @argcheck tol_check(atol, rtol)
    return EachApprox(; atol, rtol)
end

function ispossemidef(A :: Hermitian, approx :: AbstractApprox = Approx(A))
    Λ, X = eigen(A)
    idx = findfirst(isnneg, Λ)
    isnothing(idx) && return false

    Λ₊ = (@view Λ[idx : end])
    X₊ = (@view X[:, idx : end])
    A₊ = X₊ * Diagonal(Λ₊) * X₊'
    isapprox(A, A₊, approx)
end

ispossemidef(
    A :: AbstractMatrix,
    approx :: AbstractApprox = Approx(A)
) = ishermitian(A, approx) && ispossemidef(Hermitian(A), approx)

function nullify!(
    A :: AbstractArray{<: Real};
    atol = default_atol(A),
    rtol = default_rtol(atol, A),
    norm = frnorm
)
    ε = max(norm(A) * rtol, atol) / length(A)
    @inbounds for i in eachindex(A)
        A[i] = abs(A[i]) < ε ? zero(A[i]) : A[i]
    end
    A
end

function nullify!(
    A :: AbstractArray{<: Complex};
    atol = default_atol(A),
    rtol = default_rtol(atol, A),
    norm = frnorm
)
    ε = max(norm(A) * rtol, atol) / length(A)
    @inbounds for i in eachindex(A)
        p, q = real(A[i]), imag(A[i])
        A[i] = Complex(
            abs(p) < ε ? zero(p) : p,
            abs(q) < ε ? zero(q) : q
        )
    end
    A
end

isinside(x, seg) = let (l, r) = seg; l < x < r; end

# some pre-defined weights for AAA algorithm
gaussian(σ)                 = exp ∘ Base.Fix2(/, -2) ∘ Base.Fix2(^, 2) ∘ Base.Fix2(/, σ)
lorenzian(σ)                = Base.Fix1(/, σ ^ 2) ∘ Base.Fix2(+, σ ^ 2) ∘ Base.Fix2(^, 2)
box(σ)                      = Base.Fix2(isinside, (-σ, σ))
box(l, r)                   = Base.Fix2(isinside, (l, r))
add_background(val, weight) = Base.Fix1(+, val) ∘ weight

macro onlyif(val, ex)
    quote
        @argcheck $(esc(ex))
        $(esc(val))
    end
end

macro onlyif(val, ex, msg)
    quote
        @argcheck $(esc(ex)) $(esc(msg))
        $(esc(val))
    end
end

ishurwitz(A :: AbstractMatrix) = all(isneg ∘ real, eigvals(A))
ishurwitz(A :: AbstractMatrix, :: Val{false}) = all(ispos ∘ real, eigvals(A))
ishurwitz(A :: AbstractMatrix, :: Val{true}) = ishurwitz(A)

# needed for evaluation of a rational function of a matrix
Base.isnan(a :: AbstractArray) = any(isnan, a)

@inline unsafe_phase_factor(x) = (x / abs(x))
@inline phase_factor(x) = (iszero(x) ? one(x) : unsafe_phase_factor(x))

# takagi factorization
function takagi(A :: Symmetric, approx = Approx(A))
    _, _, V = svd(Matrix(A))
    C = transpose(V) * A * V

    W = let (_, Z) = schur(C)
        # real and imaginary parts of C commute
        # diagonalize them simultaneously
        # use Schur factorization for stability
        @inbounds for i in axes(Z, 2)
            col = @view Z[:, i]
            _, j = findmax(abs, col)
            col .*= unsafe_phase_factor(conj(col[j]))
        end
        @check isreal(Z, approx)
        @check isunitary(Z, approx)
        real(Z)'
    end

    U = W * transpose(V)
    d = diag(U * A * transpose(U))
    D = Diagonal(map(sqrt ∘ phase_factor ∘ conj, diag(U * A * transpose(U))))
    lmul!(D, U)
    return U, Diagonal(map(abs, d))
end
