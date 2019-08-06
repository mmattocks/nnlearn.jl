Sys.islinux() ? code_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/coded_obs_set" : code_binary = "F:\\PhD\\NGS_binaries\\nnlearn\\coded_obs_set"
Sys.islinux() ? matrix_output = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix" : matrix_output = "F:\\PhD\\NGS_binaries\\nnlearn\\BGHMM_sib_matrix"

function test_scoring(nsources=30, sourcelrange=3:70, mix_prior=.005, nsamples=3, nevals=10; threaded::Bool=true)
    @info "Loading coded observation set and offsets..."
    (obs, offsets) = deserialize(code_binary)
    position_start=141

    @info "Assembling sources..."
    source_priors = nnlearn.assemble_source_priors(nsources, [], sourcelrange)
    sources       = nnlearn.init_logPWM_sources(source_priors,sourcelrange)
    mix           = nnlearn.init_mixing_matrix(mix_prior,size(obs)[2],nsources)

    if !threaded
        @info "Benchmarking unthreaded scoring..."    
        @benchmark (ut_score_sources($sources, $obs, $position_start, $offsets, $mix)) samples=nsamples evals=nevals seconds=3600
    else
        @info "Benchmarking threaded scoring..."
        @benchmark (nnlearn.score_sources($sources, $obs, $position_start, $offsets, $mix)) samples=nsamples evals=nevals seconds=3600
    end
end

function test_likelihood(nsources=30, sourcelrange=3:70, mix_prior=.005, nsamples=3, nevals=1; threaded::Bool=true)
    @info "Setup..."
    position_start=141
    @info "Loading coded observation set and offsets..."
    (obs, offsets) = deserialize(code_binary)
    @info "Loading BGHMM likelihood matrix binary..."
    bg_scores = deserialize(matrix_output)
    @info "Assembling sources..."
    source_priors = nnlearn.assemble_source_priors(nsources, [], sourcelrange)
    sources       = nnlearn.init_logPWM_sources(source_priors,sourcelrange)
    mix           = nnlearn.init_mixing_matrix(.005,size(obs)[2],nsources)
    @info "Scoring sources..."
    score_mat, source_bitindex, source_wmls = nnlearn.score_sources(sources, obs, position_start, offsets, mix)

    if !threaded
        @info "Benchmarking unthreaded..."    
        @benchmark (ut_like($sources, $score_mat, $source_bitindex, $position_start, $bg_scores, $mix, $source_wmls)) samples=nsamples evals=nevals seconds=3600
    else
        @info "Benchmarking threaded..."
        @benchmark (nnlearn.IPM_likelihood($sources, $score_mat, $source_bitindex, $position_start, $bg_scores, $mix, $source_wmls)) samples=nsamples evals=nevals seconds=3600
    end
end

function ut_score_sources(sources::Vector{Tuple{Matrix{Float64},Int64}}, observations::Matrix{Int64}, position_start::Int64,  offsets::Vector{Int64}, mix_matrix::BitMatrix; revcomp=true)
    (T, O) = size(observations); T=T-1 #last position is indexing 0
    S      = length(sources)

    scores = Matrix{Matrix{Float64}}(undef,(O,S)) #o x s array of score matrices tupled with wml
    source_bitindex::BitArray = falses(T, S, O)
    source_wmls = zeros(Int64, S)

    indices     = findall(mix_matrix)
    for idx in indices
        o,s     = idx[1],idx[2]
        source  = sources[s][1] #get the PWM from the source tuple
        wml     = size(source)[1] #weight matrix length
        source_start= maximum([position_start-wml+1, offsets[o]+1]) #start scanning PWM at the farthest 3' of (position start - the WML +1, ie first emission base scored is the first position base) or (sequence offset + 1)- the latter for scaffold boundary edge case
        source_stop = T-wml+1

        scores[o,s] = nnlearn.score_source(observations[:,o], source, source_start, source_stop, revcomp) #get the scores for this oxs
        source_bitindex[(source_start+wml-1):T,s,o] .= true #set bitindex
        source_wmls[s] = wml
    end

    return scores, source_bitindex, source_wmls #score_mat cache of size matrix emission log_probabilities
end

function ut_like(sources::Vector{Tuple{Matrix{Float64},Int64}}, score_mat::Matrix{Matrix{Float64}}, source_bitindex::BitArray, position_start::Int64, bg_scores::AbstractArray{Float64}, mix::BitMatrix, source_wmls::Vector{Int64}, revcomp::Bool=true)
    revcomp ? log_motif_expectation = log(0.5 / size(bg_scores)[1]) : log_motif_expectation = log(1 / size(bg_scores)[1])#log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise

    (T,O) = size(bg_scores)

    obs_lhs=zeros(O)

    for o in 1:O
        obs_source_indices = findall(mix[o,:])
        if length(obs_source_indices) > 0
            obs_source_bitindex = source_bitindex[:,mix[o,:],o]
            obs_cardinality = length(obs_source_indices)
            obs_cardinality > 0 ? cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality)) : cardinality_penalty = 0.0
            
            obs_lhs[o] = nnlearn.weave_scores(o, T, bg_scores, score_mat, obs_source_indices, position_start, obs_source_bitindex, source_wmls, log_motif_expectation, cardinality_penalty, revcomp)
        end
    end

    return (MS_HMMBase.log_prob_sum(obs_lhs), rand()) #2nd value in tuple is arbitrary random marker to allow breaking of lh ties in a model ensemble; ensures that our nesting can be accomplished in strictly decreasing order on the basis of the 2nd value in the case that lhs are identical btw models with different Θ
end