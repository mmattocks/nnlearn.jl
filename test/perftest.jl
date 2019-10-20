using nnlearn,Serialization,Random,Test

code_binary = "/bench/PhD/NGS_binaries/nnlearn/coded_obs_set"
matrix_output = "/bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix"

@testset "numerical equiv" begin
    @info "Setup..."
    nsources=30; sourcelrange=3:70; mix_prior=.05
    Random.seed!(1)
    @info "Loading coded observation set..."
    obs = deserialize(code_binary)
    obs=Array(transpose(obs))
    obsl = [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]]

    @info "Loading BGHMM likelihood matrix binary..."
    bg_scores = deserialize(matrix_output)
    @info "Assembling sources..."
    source_priors = nnlearn.assemble_source_priors(nsources, Vector{Matrix{Float64}}(), 0.0, sourcelrange)
    sources       = nnlearn.init_logPWM_sources(source_priors,sourcelrange)
    mix           = nnlearn.init_mixing_matrix(mix_prior,size(obs)[2],nsources)
    @test isapprox(nnlearn.og_scoring(sources,obs,obsl,bg_scores,mix),nnlearn.perf_scoring(sources,obs,obsl,bg_scores,mix,[size(source[1])[1] for source in sources]))
end
