# Auxiliary functions

@inline ispos(x :: Real) = (x > zero(x))
@inline isneg(x :: Real) = (x < zero(x))
@inline isnpos(x :: Real) = !ispos(x)
@inline isnneg(x :: Real) = !isneg(x)

isinside(x, seg) = let (l, r) = seg; l < x < r; end

# function which always returns a constant
struct ConstFun{T} <: Function
    val :: T
end

(f :: ConstFun)(:: Any...) = f.val

# some pre-defined weights for AAA algorithm
gaussian(σ)                 = exp ∘ Base.Fix2(*, -0.5) ∘ Base.Fix2(^, 2) ∘ Base.Fix2(/, σ)
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
ishurwitz(A :: AbstractMatrix, :: Val{true}) = all(ispos ∘ real, eigvals(A))
ishurwitz(A :: AbstractMatrix, :: Val{false}) = ishurwitz(A)

# needed for evaluation of a rational function of a matrix
Base.isnan(a :: AbstractArray) = any(isnan, a)


deep_eltype(:: Type{T}) where {T} = let ET = eltype(T);
    T == ET ? T : deep_eltype(ET)
end
deep_eltype(:: T) where {T} = deep_eltype(T)
