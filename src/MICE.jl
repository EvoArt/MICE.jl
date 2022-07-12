module MICE
using Turing, Optim, StatsBase
import Base.rand
include.("models.jl","types.jl","impute.jl")
# Write your package code here.
export infer_pos_normal,infer_neg_normal,infer_zero_normal,infer_normal,infer_poisson,rand
end
