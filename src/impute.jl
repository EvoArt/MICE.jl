# function rand(m::Normal_Model)
#     μ = m.α .+ sum(m.β' .* (m.x .- m.transform[1]') ./ m.transform[2]',dims = 2)
#     rand.(Normal.(μ,m.σ))
# end


# function rand(m::Pos_Normal_Model)
#     μ = m.α .+ sum(m.β' .* (m.x .- m.transform[1]') ./ m.transform[2]',dims = 2)
#     #μ = m.α .+ sum(m.β' .* (m.x .- m.transform[1]') ,dims = 2)
#     rand.(truncated.(Normal.(μ,m.σ),0,Inf))
# end

# function rand(m::Neg_Normal_Model)
#     μ = m.α .+ sum(m.β' .* (m.x .-m.transform[1]') ./ m.transform[2]',dims = 2)
#     rand.(truncated.(Normal.(μ,m.σ),-Inf,0))
# end

# function rand(m::Zero_Normal_Model)
#         p = rand.(BernoulliLogit.(m.αₚ .+ sum(m.βₚ' .* (m.x .-m.transform[1]') ./ m.transform[2]',dims = 2)))
#         μ = m.α .+ sum(m.β' .* (m.x .-m.transform[1]') ./ m.transform[2]',dims = 2)
#         q = rand.(truncated.(Normal.(μ,m.σ),0,Inf))

#         p .* q
# end

# function rand(m::Poisson_Model)
#     λ = m.α .+ sum(m.β' .* (m.x .-m.transform[1]') ./ m.transform[2]',dims = 2)
#     #λ = m.α .+ sum(m.β' .* m.x ,dims = 2)
#     rand.(Poisson.(exp.(clamp.(λ,-Inf,m.max_val))))
# end


# function impute!(x)
#     x .= rand(m)  
# end

function naieve_impute(x::Array)
    X = copy(x)
    for col in eachcol(X)
        col[ismissing.(col)] .= median(skipmissing(col))
    end
    Array{Float64}(X), (1:size(X)[2])[vec(sum(ismissing.(x), dims =1) .> 0)]
end

function get_inference_mods(x)
    mods = []
    for i in 1:size(x)[2]
        y = skipmissing(x[:,i])
        try
            Int.(y)
            if (sum(y .==0) /length(y)) > 0.1
                push!(mods,infer_zero_poisson)
            elseif (sum(y .==0) > 1) & (mean(y[y .>0]) >10)
                push!(mods,infer_zero_poisson)
            else
                push!(mods,infer_poisson)
            end
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
function impute(x::Array, rounds::Int = 10; lambda = :cv)
    _,n = size(x)
    inference_mods = get_inference_mods(x)
    X,inds = naieve_impute(x) 
    n_tasks = rounds*length(inds)
    p = Progress(n_tasks, 1)
    task = 0
    for round in 1:rounds
        #println(X[1:15,:])
        for i in inds
            task+=1
            m = inference_mods[i](X[:,1:n .!=i],x[:,i],lambda = [lambda])
            #X[m.y,i] .=rand(m)
            if typeof(m) == Float64
                X[ismissing.(x[:,i]),i] = m
            else
                X[ismissing.(x[:,i]),i] .= m
            end
            #X[m.y,i] .=predict(m,X[m.y,1:n .!=i])
            update!(p, task)
        end
    end
    X
end
