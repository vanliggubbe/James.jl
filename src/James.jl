module James

import ArgCheck: @argcheck, @check
import MatrixEquations: arec, lyapc
import IsApprox: AbstractApprox, Approx, EachApprox, isunitary, ishermitian, issymmetric
import LinearAlgebra: Symmetric, Hermitian, Diagonal, norm as frnorm, opnorm, I, mul!, schur, eigvals, eigen, svd, svd!, rank, nullspace, isposdef, diag, lmul!, det
import Xsum: xsum
import BlockArrays: BlockedArray, BlockArray, Block, undef_blocks

include("utils.jl")
include("pencil.jl")
include("aaa.jl")
include("bsd.jl")
include("bose.jl")
include("embedding.jl")

export bose_factor
export bcf_factor
export FactorizedBSD
export Jame
export kossakovski, hamiltonian, drift, symplform, ndof
export ispossemidef

end
