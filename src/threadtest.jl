dict = Dict{Int64,Vector{Float64}}()
nentries = 5e5

Threads.@threads for n in 1:nentries
    dict[n] = [rand() for i in 1:100]
end

@assert length(dict) == nentries