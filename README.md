# James.jl

Construction of Markovian embedding for a generic continuous-variable open quantum system. Generalization of work [arXiv:2604.06466](https://arxiv.org/abs/2604.06466) with hybrid classical-quantum master equation ([PRA **107**, 062206 (2023)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.062206)) instead of Lindblad equation.

Library is under intense construction! Many things may and will change.

## Current functional

To construct a `Jame` (just another Markovian embedding), one needs bath spectral density $\boldsymbol J(\omega)$ and temperature $T$.
Bath spectral density is specified by three real entry matrices `L`, `M`, and `R` as

$$\boldsymbol J(\omega) = \omega \boldsymbol L^\mathrm{T} (\omega + i \boldsymbol M)^{-1} \boldsymbol R \boldsymbol R^\mathrm{T} \left(\omega - i \boldsymbol M^\mathrm T\right)^{-1} \boldsymbol L .$$

All the eigenvalues of `M` must have positive real parts.
```julia
# generate random matrices
L = randn(10, 3)
M = randn(10, 10)
R = randn(10, 3)

# make (-M) Hurwitz-stable
M -= 1.1 * I * real(first(eigvals(M)))

J = FactorizedBSD(L, M, R)

# construct Markovian embedding
# second argument is temperature of the bath
me = Jame(J, 2)

# print HME matrices
display(kossakovski(me))
display(hamiltonian(me))
display(drift(me))
```
