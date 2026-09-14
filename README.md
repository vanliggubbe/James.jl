# Jame.jl (Just another Markovian embedding)

Construction of Markovian embedding for a generic continuous-variable open quantum system. Generalization of work [arXiv:2604.06466](https://arxiv.org/abs/2604.06466) with hybrid classical-quantum master equation ([PRA **107**, 062206 (2023)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.062206)) instead of Lindblad equation.

Library is under intense construction! Many things may and will change.

## Current functional

Bath spectral density \\(\boldsymbol J(\omega)\\) is specified by three real entry matrices \\(\boldsymbol L\\), \\(\boldsymbol M\\), and \\(\boldsymbol R\\) as
\\[ \boldsymbol J(\omega) = \omega \boldsymbol L^\mathsf{T} (\omega + i \boldsymbol M)^{-1} \boldsymbol R \boldsymbol R^\mathsf{T} (\omega - i \boldsymbol M^\mathsf T)^{-1} \boldsymbol L . \\]
where all the eigenvalues of \\(\boldsymbol M\\) have real parts. Then, for a given temperature \\(T\\), 
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
me = MarkovianEmbedding(J, 2)

# print HME matrices
display(kossakovski(me))
display(hamiltonian(me))
display(drift(me))
```
