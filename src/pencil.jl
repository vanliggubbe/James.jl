struct RationalPencil{
    TP, TC <: Number, TL, TM, TR
}
    poly :: TP
    c :: TC
    L :: TL
    M :: TM
    R :: TR
end

(f :: RationalPencil)(ω) = sum(
    a * ω ^ (i - 1) for (i, a) in enumerate(f.poly);
    init = f.L' * ((ω * I - f.c * f.M) \ f.R)
)

Base.eltype(:: Type{RationalPencil{TP, TC, TL, TM, TR}}) where {TP, TC, TL, TM, TR} = promote_type(
    deep_eltype(TP), TC, eltype(TL), eltype(TM), eltype(TR)
)
