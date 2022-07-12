
struct Normal_Model
    x::SubArray
    α::Float64
    β::Vector{Float64}
    σ::Float64
    transform::Tuple{Vector{Float64},Vector{Float64}}
end
struct Pos_Normal_Model
    x::SubArray
    α::Float64
    β::Vector{Float64}
    σ::Float64
    transform::Tuple{Vector{Float64},Vector{Float64}}
end
struct Neg_Normal_Model
    x::SubArray
    α::Float64
    β::Vector{Float64}
    σ::Float64
    transform::Tuple{Vector{Float64},Vector{Float64}}
end
struct Zero_Normal_Model
    x::SubArray
    α::Float64
    β::Vector{Float64}
    σ::Float64
    αₚ::Float64
    βₚ::Vector{Float64}
    transform::Tuple{Vector{Float64},Vector{Float64}}

end
struct Poisson_Model
    x::SubArray
    α::Float64
    β::Vector{Float64}
    transform::Tuple{Vector{Float64},Vector{Float64}}
end
