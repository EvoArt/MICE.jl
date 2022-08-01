

# function muladd!(c,z,y,x)
#     mul!(c,z,y)
#     c .+= x
# end

# @model function poisson(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
#                             n= size(x)[1],m= size(x)[2], ::Type{T}=Float64) where {T}
#     λ = Vector{T}(undef,n)
#     α ~ Normal(log(mean(y)),max(median(y),1))
#     β ~ filldist(Cauchy(0,1/m),m)  

#     muladd!(λ,X,β,α)
#      for i in 1:n
#         #y[i] ~ Poisson(exp(λ[i]))
#         Turing.@addlogprob! poislogpdf(exp(λ[i]),y[i]) 
#     end
#     #y .~ Poisson.(exp.(λ))
#     #println(exp.(λ))
# end

# @model function normal(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
#     n= size(x)[1],m= size(x)[2], ::Type{T}=Float64) where {T}
#     μ = Vector{T}(undef,n)

#     α ~ Normal(mean(y),abs(2mean(y)))
#     β ~ filldist(Cauchy(0,1/m),m)  
#     σ ~ Exponential(std(y) +0.1)
#     muladd!(μ,X,β,α)
#      for i in 1:n
#         y[i] ~ Normal(μ[i],σ)
#     end
# end

# @model function pos_normal(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
#     n= size(x)[1],m= size(x)[2], ::Type{T}=Float64) where {T}
#     μ = Vector{T}(undef,n)
#     sig ~ Exponential(0.00000001)
#     α ~ Normal(mean(y),2mean(y))
#     β ~ filldist(Cauchy(0,1/m),m)  
#     σ ~ Exponential()
#     muladd!(μ,X,β,α)
#     #μ = vec(sum(x .* β', dims=2) .+α)
#      for i in 1:n
#         #y[i] ~ truncated(Normal(μ[i],σ);lower = 0.0)
#         y[i] ~ Normal(μ[i],σ)
#     end
# end

# @model function neg_normal(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
#     n= size(x)[1],m= size(x)[2], ::Type{T}=Float64) where {T}
#     μ = Vector{T}(undef,n)

#     α ~ Normal(mean(y),abs(2mean(y)))
#     β ~ filldist(Cauchy(0,1/m),m)  
#     σ ~ Exponential(std(y)+0.1)
#     muladd!(μ,X,β,α)
#      for i in 1:n
#         y[i] ~ Normal(μ[i],σ)
#     end
# end

# @model function zero_normal(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
#                             y_bool = y .>0,  n= size(x)[1],m= size(x)[2], ::Type{T}=Float64) where {T}
#     μ = Vector{T}(undef,sum(y_bool))
#     p = Vector{T}(undef,n)
#     bool_inds = (1:n)[y_bool]
#     x_bool = X[y_bool,:]

#     α1 ~ Normal(mean(y[y_bool]),2mean(y[y_bool]))
#     α2 ~ Normal(mean(y_bool),2mean(y_bool))
#     β1 ~ filldist(Cauchy(0,1/m),m)  
#     β2 ~ filldist(Normal(),m)  
#     σ ~ Exponential(std(y[y_bool])+0.1)

#     muladd!(μ,x_bool,β1,α1)
#     muladd!(p,X,β2,α2)

#     for (i,j) in enumerate(bool_inds)
#         y[j] ~ Normal(μ[i],σ)
#     end

#     for i in 1:n
#         y_bool[i] ~ BernoulliLogit(p[i])
#     end
# end

# @model function prop(x,y, X = (x .- mean(x,dims=1)) ./ max.(std(x,dims=1),0.000001),
#                      n= size(x)[1],m= size(x)[2])

#     #α ~ filldist(Normal(0,1),size(y)[2])
#     β ~ filldist(Normal(0,10),size(y)[2],m)  

#     for i in 1:size(y)[1]
#         y[i,:] ~ Dirichlet(abs.(β * x[i,:]) )
#     end
# end


function infer_normal(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y); lambda = [:cv])
    #model = normal(x[.! missing_inds,inds],y[.! missing_inds])
    #map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    #pars = coef(map_estimate)
    #α = pars[:α]
    #β = vec(pars[2:end-1])
    #σ = pars[end]
    #transform = Tuple([vec(mean(x[.! missing_inds,inds],dims=1)), vec(max.(std(x[.! missing_inds,inds],dims=1),0.000001))])
    #Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,transform )
    if lambda[1] == :cv
        cv = glmnetcv(x[.! missing_inds,inds],float.(y[.! missing_inds]))
        lambda = [cv.lambda[argmin(cv.losses)]]
    end
    norm_fit = glmnet(x[.! missing_inds,inds],float.(y[.! missing_inds]), lambda = lambda)
    preds = GLMNet.predict(norm_fit,x[.! missing_inds,inds])
    σ = √mean((preds .- float.(y[.! missing_inds])) .^2)
    rand.(Normal.(GLMNet.predict(norm_fit,x[missing_inds,inds]),σ))
end
function infer_pos_normal(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y); lambda = [:cv])
    #model = pos_normal(x[.! missing_inds,inds],float.(y[.! missing_inds]))
    #map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    #pars = coef(map_estimate)
    #α = pars[:α]
    #β = vec(pars[3:end-1])
    #σ = pars[end]
    #println([α maximum(β) minimum(β) mean_y mean(skipmissing(y)) std(skipmissing(y))])
    #transform = Tuple([vec(mean(x[.! missing_inds,inds],dims=1)), vec(max.(std(x[.! missing_inds,inds],dims=1),0.000001))])
    #Pos_Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,transform )
    if lambda[1] == :cv
        cv = glmnetcv(x[.! missing_inds,inds],float.(y[.! missing_inds]))
        lambda = [cv.lambda[argmin(cv.losses)]]
    end
    norm_fit = glmnet(x[.! missing_inds,inds],float.(y[.! missing_inds]), lambda = lambda)
    preds = GLMNet.predict(norm_fit,x[.! missing_inds,inds])
    σ = √mean((preds .- float.(y[.! missing_inds])) .^2)
    rand.(truncated.(Normal.(GLMNet.predict(norm_fit,x[missing_inds,inds]),σ),0.0,Inf))
end
function infer_neg_normal(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y); lambda = [:cv])
    #model = neg_normal(x[.! missing_inds,inds],y[.! missing_inds])
    #map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    #pars = coef(map_estimate)
    #α = pars[:α]
    #β = vec(pars[2:end-1])
    #σ = pars[end]
    #transform = Tuple([vec(mean(x[.! missing_inds,inds],dims=1)), vec(max.(std(x[.! missing_inds,inds],dims=1),0.000001))])
    #Neg_Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,transform )
    if lambda[1] == :cv
        cv = glmnetcv(x[.! missing_inds,inds],float.(y[.! missing_inds]))
        lambda = [cv.lambda[argmin(cv.losses)]]
    end
    norm_fit = glmnet(x[.! missing_inds,inds],float.(y[.! missing_inds]), lambda = lambda)
    preds = GLMNet.predict(norm_fit,x[.! missing_inds,inds])
    σ = √mean((preds .- float.(y[.! missing_inds])) .^2)
    rand.(truncated.(Normal.(GLMNet.predict(norm_fit,x[missing_inds,inds]),σ),-Inf,0.0))
end
function infer_zero_normal(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y); lambda = [:cv])
   # model = zero_normal(x[.! missing_inds,inds],y[.! missing_inds])
   # map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
   # pars = coef(map_estimate)
    
    #α = pars[1]
    #αₚ = pars[2]
    #β = vec(pars[3:2+length(inds)])
    #β = vec(pars[2:1+length(inds)])
    #βₚ = vec(pars[3+length(inds):end-1])
    #βₚ = vec(pars[2+length(inds):end-1])
    #σ = pars[end]
    #transform = Tuple([vec(mean(x[.! missing_inds,inds],dims=1)), vec(max.(std(x[.! missing_inds,inds],dims=1),0.000001))])
    #Zero_Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,αₚ,βₚ,transform)
    lambda_norm,lambda_bin = lambda,lambda
    present_y = float.(y[.! missing_inds])
    present_x = x[.! missing_inds,inds]
    norm_mask = present_y .>0
    if lambda[1] == :cv
        cv = glmnetcv(present_x[norm_mask,:],present_y[norm_mask])
        lambda_norm = [cv.lambda[argmin(cv.losses)]]
        cv = glmnetcv(x[.! missing_inds,inds],Array{Int}(hcat(float.(y[.! missing_inds]) .==0,float.(y[.! missing_inds]) .>=0)),Binomial())
        lambda_bin = [cv.lambda[argmin(cv.losses)]]
    end
    norm_fit = glmnet(present_x[norm_mask,:],present_y[norm_mask], lambda = lambda_norm)
    preds = GLMNet.predict(norm_fit,present_x[norm_mask,:])
    σ = √mean((preds .- present_y[norm_mask]) .^2)
    norms = rand.(truncated.(Normal.(GLMNet.predict(norm_fit,x[missing_inds,inds]),σ),0.0,Inf))
    bin_fit = glmnet(x[.! missing_inds,inds],Array{Int}(hcat(float.(y[.! missing_inds]) .==0,float.(y[.! missing_inds]) .>=0)),Binomial(), lambda = lambda_bin)
    berns = rand.(Bernoulli.(logistic.(GLMNet.predict(bin_fit,x[missing_inds,inds]))))
    norms .* berns
end

function infer_zero_poisson(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y); lambda = [:cv])
    # model = zero_normal(x[.! missing_inds,inds],y[.! missing_inds])
    # map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    # pars = coef(map_estimate)
     
     #α = pars[1]
     #αₚ = pars[2]
     #β = vec(pars[3:2+length(inds)])
     #β = vec(pars[2:1+length(inds)])
     #βₚ = vec(pars[3+length(inds):end-1])
     #βₚ = vec(pars[2+length(inds):end-1])
     #σ = pars[end]
     #transform = Tuple([vec(mean(x[.! missing_inds,inds],dims=1)), vec(max.(std(x[.! missing_inds,inds],dims=1),0.000001))])
     #Zero_Normal_Model(view(x,missing_inds,inds),missing_inds,α,β,σ,αₚ,βₚ,transform)
    lambda_norm,lambda_bin = lambda,lambda
     present_y = float.(y[.! missing_inds])
     present_x = x[.! missing_inds,inds]
     norm_mask = present_y .>0
     if lambda[1] == :cv
        cv = glmnetcv(present_x[norm_mask,:],present_y[norm_mask])
        lambda_norm = [cv.lambda[argmin(cv.losses)]]
        cv = glmnetcv(x[.! missing_inds,inds],Array{Int}(hcat(float.(y[.! missing_inds]) .==0,float.(y[.! missing_inds]) .>=0)),Binomial())
        lambda_bin = [cv.lambda[argmin(cv.losses)]]
    end

     mod_fit = glmnet(present_x[norm_mask,:],present_y[norm_mask], Poisson(),lambda = lambda_norm)

     rand.(Poisson.(exp.(GLMNet.predict(mod_fit,x[missing_inds,inds]))))
     bin_fit = glmnet(x[.! missing_inds,inds],Array{Int}(hcat(float.(y[.! missing_inds]) .==0,float.(y[.! missing_inds]) .>=0)),Binomial(), lambda = lambda_bin)
     berns = rand.(Bernoulli.(logistic.(GLMNet.predict(bin_fit,x[missing_inds,inds]))))
     norms .* berns
 end
function infer_poisson(x,y,inds =1:size(x)[2], missing_inds = ismissing.(y); lambda = [:cv])
    #model = poisson(x[.! missing_inds,inds],y[.! missing_inds])
    #map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    #pars = coef(map_estimate)
    #α = pars[:α]
    #β = vec(pars[2:end])
    #transform = Tuple([vec(mean(x[.! missing_inds,inds],dims=1)), vec(max.(std(x[.! missing_inds,inds],dims=1),0.000001))])
    #Poisson_Model(view(x,missing_inds,inds),missing_inds,α,β,transform ,log(maximum(y[.! missing_inds])))
    if lambda[1] == :cv
        cv = glmnetcv(x[.! missing_inds,inds],float.(y[.! missing_inds]))
        lambda = [cv.lambda[argmin(cv.losses)]]
    end
    mod_fit = glmnet(x[.! missing_inds,inds],float.(y[.! missing_inds]),Poisson(), lambda = lambda)
    rand.(Poisson.(exp.(GLMNet.predict(mod_fit,x[missing_inds,inds]))))
end

function infer_prop(x,y, missing_inds)
    model = prop(x[.! missing_inds,inds],y[.! missing_inds])
    map_estimate = optimize(model,MAP(),Optim.Options(allow_f_increases=true))
    pars = coef(map_estimate)
    α = pars[:α]
    β = vec(pars[2:end])
    transform = Tuple([vec(mean(x[.! missing_inds,inds],dims=1)), vec(max.(std(x[.! missing_inds,inds],dims=1),0.000001))])
    prop_Model(view(x,missing_inds,inds),missing_inds,α,β,transform )
end
