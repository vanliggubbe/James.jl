module Jame

using LinearAlgebra
using ArgCheck
import MatrixEquations: arec, lyapc
import KahanSummation: sum_kbn

include("utils.jl")
include("aaa.jl")
include("bsd.jl")
include("bose.jl")
include("embedding.jl")

export ConstFun, gaussian, lorenzian, box, add_background
export bose_factor
export bcf_factor
export CausalBSD, FactorizedBSD
export MarkovianEmbedding
export diffusion, hamiltonian, drift, symplform, ndof


end
