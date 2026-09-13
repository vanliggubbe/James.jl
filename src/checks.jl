function _ispossemidef(A, atol, rtol, norm)
    # find projector on negative part
    λ, X = eigen(A)
    idx = findall(isneg, λ)
    N = X[:, idx] * Diagonal(λ[idx]) * X[:, idx]'

    n_N = norm(N)
    n_A = norm(A)
    @check norm_check(N, n_N) "Invalid norm"
    @check norm_check(A, n_A) "Invalid norm"

    return n_N < max(atol, rtol * n_A)
end

function _ispossemidef(A, atol, rtol, :: typeof(norm))
    λ = eigvals(A)
    idx = findall(isneg, λ)
    return norm(λ[idx]) < max(atol, rtol * norm(λ))
end

function _ispossemidef(A, atol, rtol, :: typeof(opnorm))
    λ = eigvals(A)
    return first(λ) > -max(atol, rtol * abs(last(λ)))
end

function ispossemidef(
    A :: Union{Symmetric{<: Real}, Hermitian{<: Number}};
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(atol, A),
    norm :: Function = norm
)
    @argcheck tol_check(atol, rtol)
    _ispossemidef(A, atol, rtol, norm)
end

isalmosthermitian(
    A;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(atol, A),
    norm :: Function = norm
) = isapprox(A, A'; atol, rtol, norm)
    
isalmostsymmetric(
    A;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(atol, A),
    norm :: Function = norm
) = isapprox(A, transpose(A); atol, rtol, norm)

ispossemidef(
    A :: AbstractMatrix;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(atol, A),
    norm :: Function = norm
) = isalmosthermitian(A; atol, rtol) && ispossemidef(Hermitian(A + A') / 2; atol, rtol, norm)

isalmostreal(
    A;
    norm :: Function = norm,
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(atol, A)
) = deep_eltype(A) <: Real ? true : isapprox(A, conj(A); atol, rtol, norm)

ispossemidef(A :: Real; atol :: Real = default_atol(A), kwargs...) = (A >= -atol)
ispossemidef(
    A :: Complex;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(atol, A)
) = (isalmostreal(A; atol, rtol) && ispossemidef(real(A); atol = min(atol, abs(A) * rtol)))

isalmostunitary(
    A :: AbstractMatrix;
    atol :: Real = default_atol(A),
    rtol :: Real = default_rtol(atol, A),
    norm :: Function = norm
) = isapprox(A * A', I; atol, rtol, norm)


