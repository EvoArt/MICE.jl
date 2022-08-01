module MICE
using Statistics, Distributions
using StatsFuns, ProgressMeter,GLMNet

include.(["types.jl","models.jl","impute.jl"])

export infer_pos_normal,infer_neg_normal,infer_zero_normal,infer_normal,infer_poisson,rand,impute
end
