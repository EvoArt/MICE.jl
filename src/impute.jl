

function rand.(x,m::Pos_Normal_Model)
    μ = m.α .+ sum(m.β' .* (m.x .- m.transform[1]') ./ m.transform[2]',dims = 2)
    rand.(truncated.(Normal.(μ,m.σ),0,Inf))
end

function rand.(x,m::Neg_Normal_Model)
    μ = m.α .+ sum(m.β' .* (x[m.inds] .-m.transform[1]') ./ m.transform[2]',dims = 2)
    rand.(truncated.(Normal.(μ,m.σ),-Inf,0))
end

function rand.(x,m::Zero_Normal_Model)
        p = rand.(BernoulliLogit.(m.αₚ .+ sum(m.βₚ' .* (x[m.inds] .-m.transform[1]') ./ m.transform[2]')))
        μ = m.α .+ sum(m.β' .* (x[m.inds] .-m.transform[1]') ./ m.transform[2]',dims = 2)
        q = rand.(truncated.(Normal.(μ,m.σ),0,Inf))

        p .* q
end

function rand.(x,m::Poisson_Model)
    λ = m.α .+ sum(m.β' .* (x[m.inds] .-m.transform[1]') ./ m.transform[2]',dims = 2)
    rand.(Poisson.(λ))
end
