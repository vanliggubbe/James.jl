# function which always returns a constant

struct ConstFun{T} <: Function
    val :: T
end

(f :: ConstFun)(:: Any) = f.val

# Auxiliary functions

@inline ispos(x :: Real) = (x > zero(x))
@inline isneg(x :: Real) = (x < zero(x))
@inline isnpos(x :: Real) = !ispos(x)
@inline isnneg(x :: Real) = !isneg(x)

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


