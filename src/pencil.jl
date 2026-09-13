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
    a * ω ^ (i - 1) for (i, a) in f.poly;
    init = f.L' * ((ω * I - f.c * f.M) \ f.R)
)
