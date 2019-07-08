mutable struct ICA_PWM_model
    name::String #designator for saving model to posterior

    sources::Vector{Tuple{Matrix{Float64},Int64}} #vector of PWM signal sources (LOG PROBABILITY!!!) tupled with an index denoting the position of the first PWM base on the prior matrix- allows us to permute length and redraw from the appropriate prior position
    source_length_limits::UnitRange{Int64} #min/max source lengths for init and permutation
    
    mixing_matrix::BitMatrix # obs x sources bool matrix
    clean_matrix::BitMatrix #sources clean matrix- used to check if recalculating the likelihood of some observation is necessary. mark true for a mix matrix position that does not require recalculation

    source_scores::SimpleSparseArray{Float64} #sparse array of (t x observation x source) indexed scores - serves as calculation cache for model

    log_likelihood::Tuple{Float64, Float64}
end

#ICA_PWM_model FUNCTIONS
ICA_PWM_model(name::String, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, position_size::Int64, offsets::Vector{Int64}, source_length_limits::UnitRange{Int64}) = init_IPM(name, source_priors,mix_prior,bg_scores,observations,position_size,offsets,source_length_limits)

#distributed dispatch
ICA_PWM_model(jobs_chan::RemoteChannel, results_chan::RemoteChannel, name::String, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, position_size::Int64, offsets::Vector{Int64}, source_length_limits::UnitRange{Int64}) = init_IPM(jobs_chan, results_chan, name, source_priors,mix_prior,bg_scores,observations,position_size,offsets,source_length_limits)

#MODEL INIT
function init_IPM(name::String, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, position_size::Int64, offsets::Vector{Int64}, source_length_limits::UnitRange{Int64})
    T,O = size(observations)
    S=length(source_priors)
    sources=init_logPWM_sources(source_priors, source_length_limits)
    mix=init_mixing_matrix(mix_prior,O,S)
    clean=falses(size(mix)) #all obs x sources start dirty- must initialise by calculating everything once
    source_scores=score_sources!(sources, observations, position_size, offsets, mix, clean)
    clean=mix #all are now clean
    log_lh = IPM_likelihood(sources, source_scores, bg_scores, mix, position_size, offsets)

   return ICA_PWM_model(name, sources, source_length_limits, mix, clean, source_scores, log_lh)
end

function init_IPM(jobs_chan::RemoteChannel, results_chan::RemoteChannel, name::String, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, position_size::Int64, offsets::Vector{Int64}, source_length_limits::UnitRange{Int64})
    T,O = size(observations)
    S=length(source_priors)
    sources=init_logPWM_sources(source_priors, source_length_limits)
    mix=init_mixing_matrix(mix_prior,O,S)
    clean=falses(size(mix)) #all obs x sources start dirty- must initialise by calculating everything once
    source_scores=score_sources!(sources, observations, position_size, offsets, mix, clean)
    clean=mix #all are now clean
    log_lh = IPM_likelihood(jobs_chan, results_chan, sources, source_scores, bg_scores, mix, position_size, offsets)

   return ICA_PWM_model(name, sources, source_length_limits, mix, clean, source_scores, log_lh)
end
                #init_IPM SUBFUNCS
                function init_logPWM_sources(prior_vector::Vector{Vector{Dirichlet{Float64}}}, source_length_limits::UnitRange{Int64})
                    srcvec = Vector{Tuple{Matrix{Float64},Int64}}()
                    prior_coord = 1
                        for (p, prior) in enumerate(prior_vector)
                            min_PWM_length=source_length_limits[1]
                            PWM_length = rand(min_PWM_length:length(prior))
                            PWM = zeros(PWM_length,4)
                            prior_coord = rand(1:length(prior)-PWM_length+1)
                            for (position, dirichlet) in enumerate(prior)
                                if position >= prior_coord
                                    sample_coord = min(position-prior_coord+1,PWM_length)
                                    PWM[sample_coord, :] = rand(dirichlet)
                                end
                            end
                            push!(srcvec, (log.(PWM), prior_coord))
                        end
                    return srcvec
                end

                function init_mixing_matrix(mix_prior, no_observations::Int64, no_sources::Int64)
                    mix_matrix = falses(no_observations, no_sources)
                    for index in eachindex(mix_matrix)
                        if rand() <= mix_prior 
                            mix_matrix[index] = true
                        end
                    end
                    return mix_matrix
                end

#LIKELIHOOD SCORING FUNCS
function score_sources!(sources::Vector{Tuple{Matrix{Float64},Int64}},  observations::AbstractArray{Int64}, position_size::Int64, offsets::Vector{Int64}, mix_matrix::BitMatrix, clean_matrix::BitMatrix=falses(size(mix_matrix)), cache_array::SimpleSparseArray=SimpleSparseArray(size(observations)[1], 2, size(observations)[2],length(sources));revcomp=true) #cache is T x strand x O X S

    S = length(sources) #total # of sources
    
    calc_matrix::BitMatrix = mix_matrix-clean_matrix #gives which obs x source combos are dirty & require calculation

    indices = findall(calc_matrix)

    score_meter = Progress(length(indices), desc="Scoring sources: ", offset=1)
    update!(score_meter, 0)
    #counter = Threads.Atomic{Int}(0)

    #Threads.@threads for idx in indices
    for (i, idx) in enumerate(indices)
        (o, s)= (idx[1],idx[2])
        T = length(observations[:,o])
        source = sources[s][1] #get the PWM from the source tuple
        wml = size(source)[1] #weight matrix length
        source_start = maximum([position_size-wml+1, offsets[o]+1]) #start scanning PWM at the farthest 3' of (position start - the WML +1, ie first emission base scored is the first position base) or (sequence offset + 1)- the latter for scaffold boundary edge case
        source_stop = T - position_size

        score_matrix = score_source(observations[:,o], source, source_start, source_stop, revcomp)
        cache_array[(source_start+wml-1):(source_stop+wml-1),:,o,s] = score_matrix
        #Threads.atomic_add!(counter, 1)
        #Threads.threadid() == 1 && update!(score_meter, counter[])
        update!(score_meter, i)
    end

    #update!(score_meter, length(indices))

    return cache_array #source_scores cache of size matrix emission log_probabilities
end

function score_source(observation::AbstractArray{Int64,1}, source::Matrix{Float64}, source_start::Int64, source_stop::Int64, revcomp::Bool=true)
    #6 allocations, ~.00005s execution on larger sources ~100 positions. seems like could be improved through broadcasting but still have to iterate over the sequence to get a WM mask, couldn't get it any faster
    revcomp ? (revsource = revcomp_pwm(source); score_matrix = zeros(source_stop-source_start+1,2)) : score_matrix = zeros(source_stop-source_start+1)

    for t in source_start:source_stop
        forward_score = 0.0
        revcomp && (reverse_score = 0.0)
        obs_stop = findnext(iszero, observation, source_start)#oversize WMLs will not access positions beyond obs size (handles 3' scaffold boundary position edge case)

        for position in 1:size(source)[1]
            score_loc = t+position-1
            if score_loc < obs_stop
                forward_score += source[position,observation[score_loc]]
                revcomp && (reverse_score += revsource[position,observation[score_loc]])
            end
        end

        forward_score == 0.0 && (forward_score = +Inf)    #handles edge case of wm prob == 1. cannot leave as 0 in sparse matrix. decoded to 0 in likelihood fn
        (revcomp && reverse_score == 0.0) && (reverse_score = +Inf)

        score_matrix[(t-source_start+1),1] = forward_score
        revcomp && (score_matrix[(t-source_start+1),2] = reverse_score)
    end
    
    return score_matrix
end
                function revcomp_pwm(pwm::Matrix{Float64}) #pwm coming in as positions x A C G T - need to reverse probs across 2nd dimension
                    return pwm[:,end:-1:1]
                end

function IPM_likelihood(sources::Vector{Tuple{Matrix{Float64},Int64}},source_scores::SimpleSparseArray{Float64},bg_scores::AbstractArray{Float64},mix::BitMatrix, position_size::Int64, offsets::Vector{Int64}, revcomp::Bool=true)
    revcomp ? log_motif_expectation = log(0.5 / size(bg_scores)[1]) : log_motif_expectation = log(1 / size(bg_scores)[1])#log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise

    source_lengths = [wm_size[1] for wm_size in size.([source for (source, PWM_index) in sources])] #iterating over source PWMs, get the length (dim 1) of each
    T = size(bg_scores)[1]
    O = size(bg_scores)[2]

    obs_lhs=zeros(O)

    weave_meter = Progress(O, desc="Weaving scores: ", offset=2)
    update!(weave_meter, 0)
    counter = Threads.Atomic{Int}(0)

    Threads.@threads for o in 1:O
        obs_source_indices = findall(mix[o,:])
        obs_cardinality = length(obs_source_indices)
        if obs_cardinality > 0
            cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality))
        else
            cardinality_penalty = 0.0
        end

        obs_lhs[o] = weave_scores(T, bg_scores[:,o], source_scores[:,:,o,:], source_lengths, log_motif_expectation, cardinality_penalty, revcomp)
        Threads.atomic_add!(counter, 1)
        Threads.threadid() == 1 && update!(weave_meter, counter[])
    end
    update!(weave_meter, O)

    return (MS_HMMBase.log_prob_sum(obs_lhs...), rand()) #2nd value in tuple is arbitrary random marker to allow breaking of lh ties in a model ensemble; ensures that our nesting can be accomplished in strictly decreasing order on the basis of the 2nd value in the case that lhs are identical btw models with different Θ
end
                function weave_scores(T::Int64, bg_score::Vector{Float64}, source_scores::SimpleSparseArray, source_lengths::Vector{Int64}, log_motif_expectation::Float64, cardinality_penalty::Float64, revcomp::Bool=true)
                    lh_matrix = zeros(T+1)

                    for i in 2:(T+1)
                        t=i-1
                        score = lh_matrix[t] + bg_score[t] + cardinality_penalty
                        #score == -Inf && println("lh_matrix_t: $(lh_matrix[t]), bg_scores: $(bg_scores[t,o]), card pen: $cardinality_penalty, lme: $log_motif_expectation")

                        t_sources = findall(!iszero, source_scores[t,1,:])
                        if length(t_sources) < 0
                            for source_index in t_sources
                                wml = source_lengths[source_index]
                                from_score = lh_matrix[i-wml]
                                source_scores[t,1,source_index] == +Inf ? emit_score = 0.0 : emit_score = source_scores[t,1,source_index] #handles edge case of WM scoring p=1
                                score = logsumexp(MS_HMMBase.log_prob_sum(score, (from_score + emit_score + log_motif_expectation)))
                                #score == -Inf && println("s: $source_index, i: $i, t: $t, o: $o, wml: $wml, from: $from_score, emit: $emit_score, lme: $log_motif_expectation")
                                if revcomp #repeat each source contribution with the revcomp score vector if required
                                    from_score = lh_matrix[i-wml]
                                    source_scores[t,2,source_index] == +Inf ? emit_score = 0.0 : emit_score = source_scores[t,2,source_index]
                                    score = logsumexp(MS_HMMBase.log_prob_sum(score, (from_score + emit_score + log_motif_expectation)))
                                end
                            end
                        end
                        lh_matrix[i] = score
                    end
                    return lh_matrix[end]
                end

#distributed dispatch
function IPM_likelihood(jobs_chan::RemoteChannel, results_chan::RemoteChannel, sources::Vector{Tuple{Matrix{Float64},Int64}},source_scores::SimpleSparseArray{Float64},bg_scores::AbstractArray{Float64},mix::BitMatrix, position_size::Int64, offsets::Vector{Int64}; revcomp::Bool=true)
    revcomp ? log_motif_expectation = log(0.5 / size(bg_scores)[1]) : log_motif_expectation = log(1 / size(bg_scores)[1])#log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise

    source_lengths = [wm_size[1] for wm_size in size.([source for (source, PWM_index) in sources])] #iterating over source PWMs, get the length (dim 1) of each
    T = size(bg_scores)[1]
    O = size(bg_scores)[2]

    obs_lhs=zeros(O)

    @showprogress 1 "Building weaving queue..." for o in 1:O
        obs_source_indices = findall(mix[o,:])
        obs_cardinality = length(obs_source_indices)
        if obs_cardinality > 0
            cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality))
        else
            cardinality_penalty = 0.0
        end
        revcomp
        put!(jobs_chan, (o, (T, bg_scores[:,o], source_scores[:,:,o,:], obs_source_indices, source_lengths, log_motif_expectation, cardinality_penalty, revcomp)))
    end

    @showprogress 1 "Weaving scores..." for o in 1:O
        wait(results_chan)
        (obs_index, result) = take!(results_chan)
        obs_lhs[obs_index] = result
    end
        
    return (MS_HMMBase.log_prob_sum(obs_lhs...), rand()) #2nd value in tuple is arbitrary random marker to allow breaking of lh ties in a model ensemble; ensures that our nesting can be accomplished in strictly decreasing order on the basis of the 2nd value in the case that lhs are identical btw models with different Θ
end

function IPM_likelihood_worker(jobs_chan::RemoteChannel, results_chan::RemoteChannel)
    while true
        wait(jobs_chan)
        o, params = take!(jobs_chan)
        put!(results_chan, (o, weave_scores(params...)))
    end
end

#MODEL PERMUTATION INTERFACE- provides basic random decorrelation.
function permute_model!(m::ICA_PWM_model, model_no::Int64, observations::Matrix{Int64}, bg_scores::Matrix{Float64}, position_size::Int64, offsets::Vector{Int64}, priors::Vector{Vector{Dirichlet{Float64}}}, permutation_moves::Int64=100, move_type_weights::Vector{Float64}=ones(3)/3, PWM_shift_dist::Distributions.Uniform=Uniform(.0001,.02)) #shift_dist is given in probability values- converted to log space in permute_source_lengths!
    mix_moves, PWM_weight_moves, PWM_length_moves = Int.(round.(permutation_moves .* move_type_weights)) #divide total number of moves between three move types by supplied weights

    mix_moves > 0 && mix_matrix_decorrelate!(m.mixing_matrix, m.clean_matrix, mix_moves)
    PWM_weight_moves > 0 && permute_source_weights!(m.sources, m.clean_matrix, PWM_weight_moves, PWM_shift_dist)
    PWM_length_moves > 0 && permute_source_lengths!(m.sources, priors, m.clean_matrix, PWM_length_moves, m.source_length_limits)

    source_scores=score_sources!(m.sources, observations, position_size, offsets, m.mixing_matrix, m.clean_matrix, m.source_scores)
    m.clean_matrix = m.mixing_matrix
    m.log_likelihood = IPM_likelihood(m.sources, m.source_scores, bg_scores, m.mixing_matrix, position_size, offsets)
    m.name = string(model_no)
end

#distributed dispatch
function permute_model!(jobs_chan::RemoteChannel, results_chan::RemoteChannel, m::ICA_PWM_model, model_no::Int64, observations::Matrix{Int64}, bg_scores::Matrix{Float64}, position_size::Int64, offsets::Vector{Int64}, priors::Vector{Vector{Dirichlet{Float64}}}, permutation_moves::Int64=100, move_type_weights::Vector{Float64}=ones(3)/3, PWM_shift_dist::Distributions.Uniform=Uniform(.0001,.02)) #shift_dist is given in probability values- converted to log space in permute_source_lengths!
    mix_moves, PWM_weight_moves, PWM_length_moves = Int.(round.(permutation_moves .* move_type_weights)) #divide total number of moves between three move types by supplied weights

    mix_moves > 0 && mix_matrix_decorrelate!(m.mixing_matrix, m.clean_matrix, mix_moves)
    PWM_weight_moves > 0 && permute_source_weights!(m.sources, m.clean_matrix, PWM_weight_moves, PWM_shift_dist)
    PWM_length_moves > 0 && permute_source_lengths!(m.sources, priors, m.clean_matrix, PWM_length_moves, m.source_length_limits)

    source_scores=score_sources!(m.sources, observations, position_size, offsets, m.mixing_matrix, m.clean_matrix, m.source_scores)
    m.clean_matrix = m.mixing_matrix
    m.log_likelihood = IPM_likelihood(jobs_chan, results_chan, m.sources, m.source_scores, bg_scores, m.mixing_matrix, position_size, offsets)
    m.name = string(model_no)
end

                #PERMUTATION SUBFUNCS
                function mix_matrix_decorrelate!(mix::BitMatrix, clean::BitMatrix, moves::Int64)
                    indices_to_flip = rand(CartesianIndices(mix),moves)
                    mix[indices_to_flip] .= .!mix[indices_to_flip]
                end

                function permute_source_weights!(sources::Vector{Tuple{Matrix{Float64},Int64}}, clean::BitArray, moves::Int64, PWM_shift_dist::Distributions.Uniform)
                    for move in 1:moves
                        source_no = rand(1:length(sources))
                        source_pos = rand(1:size(sources[source_no][1])[1])
                        pos_WM = exp.(sources[source_no][1][source_pos,:]) #leaving logspace

                        base_to_permute = rand(1:4)
                        permutation_sign = rand(-1:2:1)
                        shift_size = rand(PWM_shift_dist)
                        
                        for base in 1:4 #ACGT
                            if base == base_to_permute
                                pos_WM[base_to_permute] =
                                clamp(0, #no lower than 0 prob
                                (pos_WM[base_to_permute]          #selected PWM posn
                                + permutation_sign * shift_size), #randomly permuted by size param
                                1) #no higher than prob 1
                            else
                                size_frac = shift_size / 3 #other bases shifted in the opposite direction by 1/3 the size accumulated at the base to permute
                                pos_WM[base_to_permute] =
                                clamp(0,
                                (pos_WM[base_to_permute]
                                - permutation_sign * size_frac),
                                1)
                            end
                        end
                        pos_WM = pos_WM ./ sum(pos_WM) #renormalise to sum 1
                        MS_HMMBase.isprobvec(pos_WM) #throw error if the position WM is invalid
                        sources[source_no][1][source_pos,:] = log.(pos_WM) # if ok reassign in logspace and carry on
                        clean[:,source_no] .= 0 #mark obs with this source in their mix row as dirty
                    end
                end

                function permute_source_lengths!(sources::Vector{Tuple{Matrix{Float64},Int64}}, priors::Vector{Vector{Dirichlet{Float64}}}, clean::BitArray, moves::Int64, length_limits::UnitRange{Int64}, uninformative::Dirichlet=Dirichlet([.25,.25,.25,.25]))
                    for move in 1:moves
                        source_no = rand(1:length(sources))
                        source_PWM, prior_idx = sources[source_no]
                        source_length = size(source_PWM)[1]
                        prior = priors[source_no]

                        #randomly chose to extend or contract a source, but prevent an extension or contraction of a source at its length limit if the randomly chosen sign would violate it
                        permutation_sign = rand(-1:2:1)
                        if source_length + permutation_sign > length_limits[2]
                            permutation_sign = -1
                        elseif source_length + permutation_sign < length_limits[2]
                            permutation_sign = 1
                        end

                        rand() >= .5 ? permute_5_pr = true : permute_5_pr = false
                        permute_5_pr==1 ? prior_position = prior_idx - 1 :
                                            prior_position = prior_idx + length(source_PWM)
                        
                        if permutation_sign == 1 #if we're to add a position to the PWM
                            prior_position<1 || prior_position>length(prior) ? #if it's outside the prior
                                new_WM = log.(rand(uninformative)) : #the new position row is drawn from an uninformative prior
                                new_WM = log.(rand(prior[prior_position])) #else it's drawn from the appropriate position
                            permute_5_pr ? #if we're extending the 5' end of the matrix
                                (source_PWM = vcat(transpose(new_WM), source_PWM); #push the new WM to the front
                                prior_idx -= 1) :#and decrement the prior index 
                                source_PWM = vcat(source_PWM, transpose(new_WM)) #if we're exending the 3' end, push it to the end 
                        else #if we're to remove a position
                            permute_5_pr ? (source_PWM=source_PWM[2:end,:] ; prior_idx += 1) :
                                            source_PWM=source_PWM[1:end-1,:] #do it from the proper end, incrementing prior index for 5' subtraction
                        end
                        sources[source_no] = (source_PWM, prior_idx) #update the sources vector
                        clean[:,source_no] .= 0 #mark relevant mix positions dirty
                    end
                end