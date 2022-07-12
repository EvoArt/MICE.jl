module MICE
using Turing, Optim, StatsBase
import Base.rand
include.(["types.jl","models.jl","impute.jl"])
# Write your package code here.
export infer_pos_normal,infer_neg_normal,infer_zero_normal,infer_normal,infer_poisson,rand,impute
end
