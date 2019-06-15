struct ICA_PWM_model
    observations::AbstractArray{Int64,3} #t x o coded observation array. could be a view from an ensemble. do not modify in subfuncs!!
    rel_starts::Vector{Int64} #first base of nucleosome position relative to the first base of the observation sequence

    sources::Vector{Matrix{Float64}} #vector of PWM signal sources (LOG PROBABILITY!!!)
    source_length_range::UnitRange{Int64}
    
    mixing_matrix::BitMatrix # obs x sources bool matrix
    clean_matrix::BitMatrix #sources clean matrix- used to check if recalculating the likelihood of some observation is necessary. mark true for a mix matrix position that does not require recalculation

    source_scores::SimpleSparseArray{Float64} #sparse array of (t x observation x source) indexed scores - serves as calculation cache for model
    bg_scores::AbstractArray{Float64} #should be t x obs. could be a view from an ensemble. do not modify in subfuncs!!

    log_likelihood::Float64

    ICA_PWM_model(source_priors::Vector{Vector{Dirichlet}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, rel_starts::Vector{Int64}, source_length_range::UnitRange{Int64}) = init_IPM(source_priors,mix_prior,bg_scores,observations,rel_starts, source_length_range)
end


#ICA_PWM_model FUNCTIONS
function init_IPM(source_priors::Vector{Vector{Dirichlet}}, mix_prior::Float64, bg_scores::Matrix{Float64}, observations::Matrix{Int64}, rel_starts::Vector{Int64}, source_length_range::UnitRange{Int64})
    T,O = size(observations)
    S=length(source_priors)
    sources=init_pwm_sources(source_priors)
    mix=init_mixing_matrix(mix_prior,O,S)
    clean=falses(size(mix)) #all obs x sources start dirty- must initialise by calculating everything once
    pad=source_length_range[2]-1
    source_scores=score_sources!(sources, observations, rel_starts,pad,mix,clean)
    clean=mix #all are now clean
    log_lh = IPM_likelihood(sources, source_scores, bg_scores, mix, rel_starts)

   return ICA_PWM_model(observations, rel_starts, sources, source_length_range, mix, clean, source_scores, bg_scores, log_lh)
end

function init_pwm_sources(prior_vector::Vector{Vector{Dirichlet}})
    source_matrix=Vector{Matrix{Float64}}()
        for prior in prior_vector
            PWM = zeros(length(prior),4)
            for (position, dirichlet) in enumerate(prior)
                PWM[position, :] = rand(dirichlet)
            end
            push!(source_matrix, PWM)
        end
    return source_matrix
end

function init_mixing_matrix(mix_prior, no_observations::Int64, no_sources::Int64)
    mix_matrix = falses(no_observations, no_sources)
    for index in eachindex(mix_matrix)
        prior_sample = rand(1)[1]
        if prior_sample <= mix_prior 
            mix_matrix[index] = true
        end
    end
    return mix_matrix
end
 
function score_sources!(sources::Vector{Matrix{Float64}},  observations::AbstractArray{Int64}, rel_starts::Vector{Int64}, pad::Int64, mix_matrix::BitMatrix, clean_matrix::BitMatrix=falses(size(mix_matrix)), cache_array::SimpleSparseArray=SimpleSparseArray(size(observations)[1], 2, size(observations)[2],length(sources));revcomp=true) #cache is T x strand x O X S

    S = length(sources)
    
    calc_matrix = mix_matrix-clean_matrix
    for (o, s) in findall(calc_matrix)
        obsl = length(observations[:,o])

        source = sources(s)
        wml = size(s)[1]
        source_start = max(rel_starts[o]-wml,1) #initial emit score is for the first base of the position
        source_stop =obsl-wml

        revcomp ? (revsource = revcomp_pwm(source); score_matrix = zeros(source_stop-source_start,2)) : score_matrix = zeros(source_stop-source_start)
        
        for t in source_start:source_stop
            forward_score = 0.0
            revcomp && (reverse_score = 0.0)

            for position in 1:wml
                forward_score += source[position,observations[t+position-1,o]]
                revcomp && (reverse_score += revsource[position,observations[t+position-1,o]])
            end

            score_matrix[t,1] = forward_score
            revcomp && (score_matrix[t,2] = reverse_score)
        end
        cache_array[source_start+wml-1:source_stop+wml-1,:,o,s] = score_matrix
    end

    return cache_array #source_scores cache of weight matrix emission log_probabilities
end

                function revcomp_pwm(pwm::Matrix{Float64}) #pwm coming in as positions x A C G T - need to reverse probs across 2nd dimension
                    return pwm[:,end:-1:1]
                end

function IPM_likelihood(sources::Vector{Matrix{Float64}},source_scores::SimpleSparseArray{Float64},bg_scores::Matrix{Float64},mix::BitMatrix, rel_starts; revcomp=true)
    revcomp ? log_motif_expectation = log(0.5 / size(bg_scores)[1]) :  log_motif_expectation = log(1 / size(bg_scores)[1])#log_motif_expectation-nMica has 0.5 for including the reverse complement, 1 otherwise

    source_lengths = [wm_size[1] for wm_size in size.(sources)]
    T = size(bg_scores)[1] - maximum(rel_start-1) + maximum(source_lengths-1)
    O = size(bg_scores)[2]

    obs_lhs=zeros(O)

    for o in 1:O
        obs_source_indices = findall(mix[o,:])
        obs_cardinality = length(obs_source_indices)
        cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality))
        obs_max_wml = maximum(source_lengths[obs_source_indices])

        lh_matrix = zeros(T+1) 

        for t in rel_starts[o]:T+1
            score = lh_matrix[t-1,:] .+ bg_scores[t-1,o] + cardinality_penalty
            for source_index in obs_source_indices
                wml = source_lengths[source_index]
                if t > wml
                    from_score = lh_matrix[t-wml,1]
                    emit_score = source_scores[t-1,1,o,source_index]
                    score = logsumexp(MS_HMMBase.log_prob_sum(score, (from_score + emit_score + log_motif_expectation)))
                    if revcomp #repeat each source contribution with the revcomp score vector if required
                        from_score = lh_matrix[t-wml,2]
                        emit_score = source_scores[t-1,2,o,source_index]
                        logsumexp(MS_HMMBase.log_prob_sum(score, (from_score + emit_score + log_motif_expectation)))
                    end
                end
            end
            lh_matrix[t] = score
        end
        obs_lhs[o] = lh_matrix[end]
    end
    return MS.HMMBase.log_prob_sum(obs_lhs...)
end

struct Bayes_IPM_ensemble
    models::Vector{ICA_PWM_model}
    log_likelihood::Float64

    bg_scores::Matrix{Int64}
    current_score_matrix::Matrix{Int64}
    obs_array::Array{Int64}

    retain_discarded_samples::Bool
    retined_posterior_samples::Vector{ICA_PWM_model}

    Bayes_IPM_ensemble(no_models, source_priors::Vector{Vector{Dirichlet}}, mixing_prior::Float64, background_score_matrix::Matrix{Int64}, observations_array::Array{Int64}; posterior_switch::Bool=true) = Bayes_IPM_ensemble(init_I_P_ensemble(no_models, source_priors), init_mixing_matrix(mixing_prior, size(observations_array)[2], length(source_priors)), -Inf, background_score_matrix, observations_array, zeros(size(observations_array[2])), observations_array, posterior_switch, Vector{ICA_PWM_model}())
end

#Bayes_IPM_ensemble SUBFUNCTIONS
function init_IPM_ensemble(no_models::Int64, source_priors::Vector{Vector{Dirichlet}})
    ensemble = Vector{ICA_PWM_model}()
    for model in 1:no_models
        model = ICA_PWM_model(source_priors)
        push!(ensemble, model)
    end
    return ensemble
end