
@model function poisson(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
                            n= size(x)[1],m= size(x)[2])
    α ~ Normal(log(mean(y)),log(mean(y)))
    β ~ filldist(Normal(),m)  
    λ = vec(α .+ sum(X .* β', dims = 2))

    y .~ Poisson.(exp.(λ))
    #println(exp.(λ))
end

@model function normal(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
    n= size(x)[1],m= size(x)[2])
    α ~ Normal()
    β ~ filldist(Normal(),m)  
    σ ~ Exponential(std(y) +0.1)
    μ = vec(α .+ sum(X .* β', dims = 2))

    y .~ Normal.(μ,σ)
end

@model function pos_normal(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
    n= size(x)[1],m= size(x)[2])
    α ~ Normal()
    β ~ filldist(Normal(),m)  
    σ ~ Exponential(std(y) +0.1)
    μ = vec(α .+ sum(X .* β', dims = 2))

    y .~ truncated.(Normal.(μ,σ),0,Inf)
end

@model function neg_normal(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
    n= size(x)[1],m= size(x)[2])
    α ~ Normal()
    β ~ filldist(Normal(),m)  
    σ ~ Exponential(std(y)+0.1)
    μ = vec(α .+ sum(X .* β', dims = 2))

    y .~ truncated.(Normal.(μ,σ),-Inf,0)
end

@model function zero_normal(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
                            y_bool = y .>0,  n= size(x)[1],m= size(x)[2])

    α ~ filldist(Normal(),2)
    β ~ filldist(Normal(),m,2)  
    σ ~ Exponential(std(y) +0.1)
    μ = vec(α[1] .+ sum(X[y_bool,:] .* β[:,1]', dims = 2))
    p = vec(α[2] .+ sum(X .* β[:,2]', dims = 2))

    y[y_bool] .~ truncated.(Normal.(μ,σ),0,Inf)
    y_bool .~ BernoulliLogit.(p)
end

@model function prop(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
                     n= size(x)[1],m= size(x)[2])

    #α ~ filldist(Normal(0,1),size(y)[2])
    β ~ filldist(Normal(0,10),size(y)[2],m)  

    for i in 1:size(y)[1]
        y[i,:] ~ Dirichlet(abs.(β * x[i,:]) )
    end
end


function infer_normal(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y))
    model = normal(x[.! missing_inds,inds],y[.! missing_inds])
    map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    pars = coef(map_estimate)
    α = pars[:α]
    β = vec(pars[2:end-1])
    σ = pars[end]
    transform = Tuple([vec(mean(x,dims=1)), vec(max.(std(x,dims=1),0.000001))])
    Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,transform )
end
function infer_pos_normal(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y))
    model = pos_normal(x[.! missing_inds,inds],y[.! missing_inds])
    map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    pars = coef(map_estimate)
    α = pars[:α]
    β = vec(pars[2:end-1])
    σ = pars[end]
    transform = Tuple([vec(mean(x,dims=1)), vec(max.(std(x,dims=1),0.000001))])
    Pos_Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,transform )
end
function infer_neg_normal(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y))
    model = neg_normal(x[.! missing_inds,inds],y[.! missing_inds])
    map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    pars = coef(map_estimate)
    α = pars[:α]
    β = vec(pars[2:end-1])
    σ = pars[end]
    transform = Tuple([vec(mean(x,dims=1)), vec(max.(std(x,dims=1),0.000001))])
    Neg_Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,transform )
end
function infer_zero_normal(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y))
    model = zero_normal(x[.! missing_inds,inds],y[.! missing_inds])
    map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    pars = coef(map_estimate)
    α = pars[1]
    αₚ = pars[2]
    β = vec(pars[3:2+length(inds)])
    βₚ = vec(pars[3+length(inds):end-1])
    σ = pars[end]
    transform = Tuple([vec(mean(x,dims=1)), vec(max.(std(x,dims=1),0.000001))])
    Zero_Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,αₚ,βₚ,transform)
end

function infer_poisson(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y))
    model = poisson(x[.! missing_inds,inds],y[.! missing_inds])
    map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    pars = coef(map_estimate)
    α = pars[:α]
    β = vec(pars[2:end])
    transform = Tuple([vec(mean(x,dims=1)), vec(max.(std(x,dims=1),0.000001))])
    Poisson_Model(view(x,missing_inds,inds),missing_inds,α,β,transform )
end

function infer_prop(x,y, missing_inds)
    model = prop(x[.! missing_inds,inds],y[.! missing_inds])
    map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    pars = coef(map_estimate)
    α = pars[:α]
    β = vec(pars[2:end])
    transform = Tuple([vec(mean(x,dims=1)), vec(max.(std(x,dims=1),0.000001))])
    prop_Model(view(x,missing_inds,inds),missing_inds,α,β,transform )
end
