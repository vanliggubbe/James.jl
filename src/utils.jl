# Auxiliary functions

deep_eltype(:: Type{T}) where {T} = let ET = eltype(T);
    T == ET ? T : deep_eltype(ET)
end
deep_eltype(:: T) where {T} = deep_eltype(T)

@inline concat(iterators...) = Iterators.flatten(iterators)

@inline ispos(x :: Real) = (x > zero(x))
@inline isneg(x :: Real) = (x < zero(x))
@inline isnpos(x :: Real) = !ispos(x)
@inline isnneg(x :: Real) = !isneg(x)

default_atol(x) = (zero ∘ float ∘ real ∘ deep_eltype)(x)
default_rtol(x) = (sqrt ∘ eps ∘ float ∘ real ∘ deep_eltype)(x)
default_rtol(x, atol) = iszero(atol) ? default_rtol(x) : default_atol(x)
@inline tol_check(a :: Real, r :: Real) = isnneg(a) && isnneg(r) && (ispos(a) || ispos(r))

function ispossemidef(
    A :: Union{Symmetric{<: Real}, Hermitian{<: Number}};
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(A, atol)
)
    @argcheck tol_check(atol, rtol)
    λ = eigvals(A)
    return !(λ[begin] < -atol || λ[begin] < -rtol * λ[end])
end

isalmosthermitian(
    A;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(A, atol),
    norm :: Function = norm
) = isapprox(A, A'; atol, rtol, norm)
    
isalmostsymmetric(
    A;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(A, atol),
    norm :: Function = norm
) = isapprox(A, transpose(A); atol, rtol, norm)

ispossemidef(
    A :: AbstractMatrix,
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(A, atol)
) = isalmosthermitian(A; atol, rtol) && ispossemidef(Hermitian(A + A'); atol, rtol)

isalmostreal(
    A;
    norm = norm,
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(A, atol)
) = deep_eltype(A) <: Real ? true : isapprox(A, conj(A); atol, rtol, norm)

ispossemidef(A :: Real; atol :: Real = default_atol(A), kwargs...) = (A >= -atol)
ispossemidef(
    A :: Complex;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(A, atol)
) = (isalmostreal(A; atol, rtol) && ispossemidef(real(A); atol))

isalmostunitary(
    A :: AbstractMatrix;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(A, atol)
) = isapprox(A * A', I; atol, rtol)

# function which always returns a constant
struct ConstFun{T} <: Function
    val :: T
end

(f :: ConstFun)(:: Any...) = f.val

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

function ensure(x, f)
    @argcheck f(x)
    x
end

function ensure(x, f, msg)
    @argcheck f(x) msg
    x
end

ishurwitz(A :: AbstractMatrix) = all(isneg ∘ real, eigvals(A))
ishurwitz(A :: AbstractMatrix, :: Val{false}) = all(ispos ∘ real, eigvals(A))
ishurwitz(A :: AbstractMatrix, :: Val{true}) = ishurwitz(A)

# needed for evaluation of a rational function of a matrix
Base.isnan(a :: AbstractArray) = any(isnan, a)

phase_factor(x) = iszero(x) ? one(x) : (x / abs(x))
unsafe_phase_factor(x) = (x / abs(x))

# takagi factorization
function takagi(A :: Symmetric)
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
        @assert norm(imag(Z)) < norm(real(Z)) * sqrt(eps(real(eltype(Z))))
        @assert norm(Z * Z' - I) < norm(Z * Z') * sqrt(eps(real(eltype(Z))))
        real(Z)'
    end

    U = W * transpose(V)
    d = diag(U * A * transpose(U))
    D = Diagonal(map(sqrt ∘ phase_factor ∘ conj, diag(U * A * transpose(U))))
    lmul!(D, U)
    return U, Diagonal(map(abs, d))
end
