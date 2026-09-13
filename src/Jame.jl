module Jame

import ArgCheck: @argcheck, @check
import MatrixEquations: arec, lyapc
import KahanSummation: sum_kbn
import IsApprox: AbstractApprox, Approx, EachApprox, isunitary, ishermitian, issymmetric
import LinearAlgebra: Symmetric, Hermitian, Diagonal, norm as frnorm, opnorm, I, mul!, schur, eigvals, eigen, svd, svd!, rank, nullspace, isposdef, diag, lmul!
import BlockArrays: BlockedArray, BlockArray, Block, undef_blocks

include("utils.jl")
include("pencil.jl")
include("aaa.jl")
include("bsd.jl")
include("bose.jl")
include("embedding.jl")

export gaussian, lorenzian, box, add_background
export bose_factor
export bcf_factor
export CausalBSD, FactorizedBSD
export MarkovianEmbedding
export diffusion, hamiltonian, drift, symplform, ndof


end
