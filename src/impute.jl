function rand(m::Normal_Model)
    μ = m.α .+ sum(m.β' .* (m.x .- m.transform[1]') ./ m.transform[2]',dims = 2)
    rand.(Normal.(μ,m.σ))
end


function rand(m::Pos_Normal_Model)
    μ = m.α .+ sum(m.β' .* (m.x .- m.transform[1]') ./ m.transform[2]',dims = 2)
    rand.(truncated.(Normal.(μ,m.σ),0,Inf))
end

function rand(m::Neg_Normal_Model)
    μ = m.α .+ sum(m.β' .* (m.x .-m.transform[1]') ./ m.transform[2]',dims = 2)
    rand.(truncated.(Normal.(μ,m.σ),-Inf,0))
end

function rand(m::Zero_Normal_Model)
        p = rand.(BernoulliLogit.(m.αₚ .+ sum(m.βₚ' .* (m.x .-m.transform[1]') ./ m.transform[2]',dims = 2)))
        μ = m.α .+ sum(m.β' .* (m.x .-m.transform[1]') ./ m.transform[2]',dims = 2)
        q = rand.(truncated.(Normal.(μ,m.σ),0,Inf))

        p .* q
end

function rand(m::Poisson_Model)
    λ = m.α .+ sum(m.β' .* (m.x .-m.transform[1]') ./ m.transform[2]',dims = 2)
    rand.(Poisson.(exp.(λ)))
end


function impute!(x)
    x .= rand(m)  
end

function naieve_impute(x::Array)
    X = copy(x)
    for col in eachcol(X)
        col[ismissing.(col)] .= mean(skipmissing(col))
    end
    X
end

function get_inference_mods(x)
    mods = []
    for i in 1:size(x)[2]
        y = skipmissing(x[:,i])
        try
            Int.(y)
            push!(mods,infer_poisson)
        catch
            if any(y .==0.0)
                push!(mods,infer_zero_normal)
            elseif sum(y .> 0) ==0
                push!(mods,infer_neg_normal)
            elseif sum(y .< 0) ==0
                push!(mods,infer_pos_normal)
            else 
                push!(mods,infer_normal)
            end
        end
    end
    mods
end
function impute(x::Array, rounds = 10)
    _,n = size(x)
    inference_mods = get_inference_mods(x)
    X = naieve_impute(x) 
    for round in 1:rounds
        for i in 1:n
            m = inference_mods[i](X[:,1:n .!= i],x[:,i])
            X[m.y,i] .=rand(m)
        end
    end
    X
end
            
            
