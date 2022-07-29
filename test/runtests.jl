using Revise, MICE,Distributions
using Test

X = rand([missing,1.0],500,120)
x = rand(Poisson(10),500,120)
x = randn(500,120) .+20
#x[:,2] .= (x[:,1] .-20) .*0.8 .+5
#x = randn(500,120)
X .=x# hcat([x for i in 1:120]...)

for i in 1:length(X)
    if rand() > 0.7
        X[i] = missing
    end
end
Y = impute(X[:,1:21])

model1 = pos_normal(Array{Float64}(x[:,1:2]),Vector{Float64}(x[:,3]), mean(x[:,3]))
map_estimate = optimize(model1,MAP(),Optim.Options(allow_f_increases=true));
pars = coef(map_estimate)
f(x[:,1:2],pars)


f(x,p) =p[:α] .+ sum(vec(p[3:end-1])' .* x, dims = 2)
f(x[:,1:2],pars)

@time begin
    model1 = poisson(Array{Float32}(X[:,1:30]),Int.(X[:,11]))
    mp =1.0
    for i in 1:100
    map_estimate = optimize(model1,MAP(),Optim.Options(allow_f_increases=true));
    pars = coef(map_estimate)
    mp  += pars[:α]
    end
    mp
end;

@time impute(X[:,1:21]);


x2 = randn(700,100) ./5
β = randn(100) ./5
α = 4
λ = rand(700)
muladd!(λ,x2,β,α)
y2= rand.(Poisson.(exp.(λ)))
model1 = poisson(x2,y2)
@time map_estimate = optimize(model1,MAP(),Optim.Options(allow_f_increases=true));
pars = coef(map_estimate)
α2 = pars[:α]
β2 = vec(pars[2:end])
hcat(β,β2 ./ std(x2, dims = 1)')

@time impute(X[:,1:10])

Z = impute(X)

@testset "MICE.jl" begin
    # Write your tests here.
end



