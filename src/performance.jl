code_binary = "/bench/PhD/NGS_binaries/nnlearn/coded_obs_set"
matrix_output = "/bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix"

function test_IPM_score(nsources=30, sourcelrange=3:70, mix_prior=.05, nsamples=5, nevals=3; perf::Bool=true)
    @info "Setup..."
    seed!(1)
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

    if !perf
        @info "Benchmarking og..."
        @benchmark (og_scoring($sources, $obs, $obsl, $bg_scores, $mix)) samples=nsamples evals=nevals seconds=3600
    else
        @info "Benchmarking perf..."
        @benchmark (perf_scoring($sources, $obs, $obsl, $bg_scores, $mix, $([size(source[1])[1] for source in sources]))) samples=nsamples evals=nevals seconds=3600
    end
end


function og_scoring(sources::Vector{Tuple{Matrix{Float64},Int64}}, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::AbstractArray{Float64}, mix::BitMatrix, revcomp::Bool=true)
    revcomp ? log_motif_expectation = log(0.5 / size(bg_scores)[1]) : log_motif_expectation = log(1 / size(bg_scores)[1])#log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise
    O = size(bg_scores)[2]
    obs_lhs=zeros(O)
    score_mat, source_bitindex, source_wmls = nnlearn.score_sources(sources, observations, obs_lengths, mix)
    for o in 1:O
        mixview=view(mix,o,:)
        obs_source_indices = findall(mixview)
        obs_source_bitindex = view(source_bitindex,:,mixview,o)
        obs_cardinality = length(obs_source_indices) #the more sources, the greater the cardinality_penalty
        obs_cardinality > 0 ? cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality)) : cardinality_penalty = 0.0
            
        obs_lhs[o] = weave_scores(obs_lengths[o], view(bg_scores,:,o), view(score_mat,o,:), obs_source_indices, obs_source_bitindex, view(source_wmls,:), log_motif_expectation, cardinality_penalty, revcomp)
    end

    return CLHMM.lps(obs_lhs)
end


function perf_scoring(sources::Vector{Tuple{Matrix{Float64},Int64}}, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::AbstractArray{Float64}, mix::BitMatrix, source_wmls::Vector{Int64}, revcomp::Bool=true)
    revcomp ? log_motif_expectation = log(0.5 / size(bg_scores)[1]) : log_motif_expectation = log(1 / size(bg_scores)[1])#log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise
    O = size(bg_scores)[2]
    obs_lhs=Vector{Vector{Float64}}()
    nt=Threads.nthreads()
    for t in 1:nt-1
        push!(obs_lhs,zeros(Int(floor(O/nt))))
    end
    push!(obs_lhs, zeros(Int(floor(O/nt)+(O%nt))))
    
    Threads.@threads for t in 1:nt
        opt = floor(O/nt) #obs per thread
        for i in 1:Int(opt+(t==nt)*(O%nt))
            o=Int(i+(t-1)*opt)
            obsl = obs_lengths[o]
            mixview=view(mix,o,:)
            mixwmls=source_wmls[mixview]
            score_mat=score_obs_sources(sources[mixview], observations[1:obsl,o], obsl, mixwmls)
            obs_source_indices = findall(mixview)
            obs_cardinality = length(obs_source_indices) #the more sources, the greater the cardinality_penalty
            obs_cardinality > 0 ? cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality)) : cardinality_penalty = 0.0
                
            obs_lhs[t][i] =lo_weave(obsl, view(bg_scores,:,o), score_mat, obs_source_indices, mixwmls, log_motif_expectation, cardinality_penalty, revcomp)
        end
    end

    return CLHMM.lps([CLHMM.lps(obs_lhs[t]) for t in 1:nt])
end

function score_obs_sources(sources::Vector{Tuple{Matrix{Float64},Int64}}, observation::Vector{Int64}, obsl::Int64, source_wmls::Vector{Int64}; revcomp=true) #scores:OxS matrix of score matrices; source_bitindex: TxSxO bitcube indexing which T values have scores for each SxO; clean matrix: tracks indices that are clean and do not need to be recalculated in permutations
    scores=Vector{Matrix{Float64}}()

    for (s,source) in enumerate(sources)
        pwm = source[1] #get the PWM from the source tuple
        wml = source_wmls[s] #weight matrix length
        source_stop=obsl-wml+1 #stop scannng th source across the observation here

        push!(scores, nnlearn.score_source(observation, pwm, source_stop, revcomp)) #get the scores for this oxs
    end

    return scores
end

function lo_weave(obsl::Int64, bg_scores::SubArray, score_mat::Vector{Matrix{Float64}}, obs_source_indices::Vector{Int64}, source_wmls::Vector{Int64}, log_motif_expectation::Float64, cardinality_penalty::Float64,  revcomp::Bool=true)
    L=obsl+1
    lh_vec = zeros(L)#likelihood vector is one position (0 initialiser) longer than the observation
    osi_emitting = Vector{Int64}()

    @inbounds for i in 2:L #i=1 is ithe lh_vec initializing 0, i=2 is the score of the first background position (ie t=1)
        t=i-1
        score = lh_vec[i-1] + bg_scores[t] + cardinality_penalty

        if length(osi_emitting)<length(obs_source_indices)
            for n in 1:length(obs_source_indices)
                if !(n in osi_emitting)
                    t>= source_wmls[n] && (push!(osi_emitting,n))
                end
            end
        end

        for n in osi_emitting
            wml = source_wmls[n]
            from_score = lh_vec[i-wml+1] #score at the first position of the PWM
            score_array = score_mat[n]
            score_idx = t - wml + 1
            emit_score = score_array[score_idx,1] #emission score at the last position of the PWM

            score = logaddexp(score, CLHMM.lps(from_score, emit_score, log_motif_expectation))
            if revcomp #repeat each source contribution with the revcomp score vector if required
                emit_score = score_array[score_idx,2]
                score = logaddexp(score, CLHMM.lps(from_score, emit_score, log_motif_expectation))
            end
        end
        lh_vec[i] = score
    end
    return lh_vec[end]
end