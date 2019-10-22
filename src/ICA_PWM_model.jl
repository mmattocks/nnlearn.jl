mutable struct ICA_PWM_model
    name::String #designator for saving model to posterior
    sources::Vector{Tuple{Matrix{Float64},Int64}} #vector of PWM signal sources (LOG PROBABILITY!!!) tupled with an index denoting the position of the first PWM base on the prior matrix- allows us to permute length and redraw from the appropriate prior position
    source_length_limits::UnitRange{Int64} #min/max source lengths for init and permutation
    mixing_matrix::BitMatrix # obs x sources bool matrix
    log_likelihood::Float64
end

#ICA_PWM_model FUNCTIONS
ICA_PWM_model(name::String, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, source_length_limits::UnitRange{Int64}) = init_IPM(name, source_priors,mix_prior,bg_scores,observations,source_length_limits)

#MODEL INIT
function init_IPM(name::String, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, source_length_limits::UnitRange{Int64})
    T,O = size(observations)
    S=length(source_priors)
    obs_lengths=[findfirst(iszero,observations[:,o])-1 for o in 1:size(observations)[2]]
    sources=init_logPWM_sources(source_priors, source_length_limits)
    mix=init_mixing_matrix(mix_prior,O,S)
    log_lh = IPM_likelihood(sources, observations, obs_lengths, bg_scores, mix)

   return ICA_PWM_model(name, sources, source_length_limits, mix, log_lh)
end

                #init_IPM SUBFUNCS
                function init_logPWM_sources(prior_vector::Vector{Vector{Dirichlet{Float64}}}, source_length_limits::UnitRange{Int64})
                    srcvec = Vector{Tuple{Matrix{Float64},Int64}}()
                    prior_coord = 1
                        for (p, prior) in enumerate(prior_vector)
                            min_PWM_length=source_length_limits[1]
                            PWM_length = rand(min_PWM_length:length(prior)) #generate a PWM from a random subset of the prior
                            PWM = zeros(PWM_length,4)
                            prior_coord = rand(1:length(prior)-PWM_length+1) #determine what position on the prior to begin sampling from based on the PWM length
                            for (position, dirichlet) in enumerate(prior)
                                if position >= prior_coord #skip prior positions that are before the selected prior_coord
                                    sample_coord = min(position-prior_coord+1,PWM_length) #sample_coord is the position on the sampled PWM
                                    PWM[sample_coord, :] = rand(dirichlet) #draw the position WM from the dirichlet
                                    @assert HMMBase.isprobvec(PWM[sample_coord, :]) #make sure it's a valid probvec
                                end
                            end
                            push!(srcvec, (log.(PWM), prior_coord)) #push the source PWM to the source vector with the prior coord idx to allow drawing from the appropriate prior dirichlets on permuting source length
                        end
                    return srcvec
                end

                function init_mixing_matrix(mix_prior::Float64, no_observations::Int64, no_sources::Int64)
                    @assert 0.0 <= mix_prior <=1.0
                    mix_matrix = falses(no_observations, no_sources)
                    for index in eachindex(mix_matrix)
                        rand() <= mix_prior && (mix_matrix[index] = true)
                    end
                    return mix_matrix
                end

#LIKELIHOOD SCORING FUNCS
function IPM_likelihood(sources::Vector{Tuple{Matrix{Float64},Int64}}, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::AbstractArray{Float64}, mix::BitMatrix, revcomp::Bool=true, returncache::Bool=false, cache::Vector{Float64}=zeros(0), clean::Vector{Bool}=Vector(falses(size(observations)[2])))
    source_wmls=[size(source[1])[1] for source in sources]
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
            if clean[o]
                obs_lhs[t][i]=cache[o]
            else
                obsl = obs_lengths[o]
                revcomp ? log_motif_expectation = log(0.5/obsl) : log_motif_expectation = log(1/obsl)#log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise
                mixview=view(mix,o,:)
                mixwmls=source_wmls[mixview]
                score_mat=score_obs_sources(sources[mixview], observations[1:obsl,o], obsl, mixwmls, revcomp=revcomp)
                obs_source_indices = findall(mixview)
                obs_cardinality = length(obs_source_indices) #the more sources, the greater the cardinality_penalty
                obs_cardinality > 0 ? cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality)) : cardinality_penalty = 0.0
                    
                obs_lhs[t][i]=weave_scores(obsl, view(bg_scores,:,o), score_mat, obs_source_indices, mixwmls, log_motif_expectation, cardinality_penalty, revcomp)
            end
        end
    end

    returncache ? (return CLHMM.lps([CLHMM.lps(obs_lhs[t]) for t in 1:nt]), vcat(obs_lhs...)) : (return CLHMM.lps([CLHMM.lps(obs_lhs[t]) for t in 1:nt]))
end

                function score_obs_sources(sources::Vector{Tuple{Matrix{Float64},Int64}}, observation::Vector{Int64}, obsl::Int64, source_wmls::Vector{Int64}; revcomp=true) 
                    scores=Vector{Matrix{Float64}}()

                    for (s,source) in enumerate(sources)
                        pwm = source[1] #get the PWM from the source tuple
                        wml = source_wmls[s] #weight matrix length
                        source_stop=obsl-wml+1 #stop scannng th source across the observation here

                        push!(scores, nnlearn.score_source(observation, pwm, source_stop, revcomp)) #get the scores for this oxs
                    end

                    return scores
                end

                function score_source(observation::AbstractArray{Int64,1}, source::Matrix{Float64}, source_stop::Int64, revcomp::Bool=true)
                    revcomp ? (revsource = revcomp_pwm(source); score_matrix = zeros(source_stop,2)) : score_matrix = zeros(source_stop)
                    forward_score = 0.0
                    revcomp && (reverse_score = 0.0)

                    for t in 1:source_stop
                        forward_score = 0.0 #initialise scores as log(p=1)
                        revcomp && (reverse_score = 0.0)

                        for position in 1:size(source)[1]
                            score_loc = t+position-1 #score_loc is the position of the obs to be scored by PWM
                            forward_score += source[position,observation[score_loc]] #add the appropriate log PWM value from the source to the score
                            revcomp && (reverse_score += revsource[position,observation[score_loc]])
                        end

                        score_matrix[t,1] = forward_score #assign scores to the matrix
                        revcomp && (score_matrix[t,2] = reverse_score)
                    end
                    
                    return score_matrix
                end
                                function revcomp_pwm(pwm::Matrix{Float64}) #in order to find a motif on the reverse strand, we scan the forward strand with the reverse complement of the pwm, reordered 3' to 5', so that eg. an PWM for an ATG motif would become one for a CAT motif
                                    return pwm[end:-1:1,end:-1:1]
                                end


                function weave_scores(obsl::Int64, bg_scores::SubArray, score_mat::Vector{Matrix{Float64}}, obs_source_indices::Vector{Int64}, source_wmls::Vector{Int64}, log_motif_expectation::Float64, cardinality_penalty::Float64,  revcomp::Bool=true)
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

#DECORRELATION FUNCTIONS

#random, iterative permutation
function permute_model!(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64=10, permutation_moves::Int64=100, move_type_weights::Vector{Float64}=ones(3)/3, PWM_shift_dist::Distributions.Uniform=Uniform(.0001,.02)) #shift_dist is given in decimal probability values- converted to log space in permute_source_lengths!

    m.log_likelihood = -Inf; iterate = 1 #init for iterative likelihood search
    @assert length(move_type_weights) == 3
    mix_moves, PWM_weight_moves, PWM_length_moves = Int.(round.(permutation_moves .* move_type_weights)) #divide total number of moves between three move types by supplied weights
    mix_moves > length(m.mixing_matrix) && (mix_moves = length(m.mixing_matrix))
    T,O = size(observations); T=T-1
    S = length(m.sources)

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mixing_matrix, true, true)
    clean=Vector{Bool}(trues(O))

    while m.log_likelihood < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        mix_moves > 0 && mix_matrix_decorrelate!(m.mixing_matrix, mix_moves, clean)
        PWM_weight_moves > 0 && permute_source_weights!(m.mixing_matrix, m.sources, PWM_weight_moves, PWM_shift_dist, clean)
        PWM_length_moves > 0 && permute_source_lengths!(m.mixing_matrix, m.sources, priors, PWM_length_moves, m.source_length_limits, clean)
        m.log_likelihood, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mixing_matrix, true, true, cache, clean)
        iterate += 1
    end
end

                #PERMUTATION SUBFUNCS
                function mix_matrix_decorrelate!(mix::BitMatrix, moves::Int64, clean::Vector{Bool})
                    indices_to_flip = Vector{CartesianIndex{2}}()
                    for i in 1:moves
                        index_to_flip = rand(CartesianIndices(mix))
                        while index_to_flip in indices_to_flip #sampling without replacement, want to always get the requested # of indices
                            index_to_flip = rand(CartesianIndices(mix))
                        end
                        push!(indices_to_flip, index_to_flip)
                    end
                    mix[indices_to_flip] .= .!mix[indices_to_flip]
                    clean[unique([idx[1] for idx in indices_to_flip])] .= false #track dirty indices
                end

                function permute_source_weights!(mix::BitMatrix, sources::Vector{Tuple{Matrix{Float64},Int64}}, moves::Int64, PWM_shift_dist::Distributions.Uniform, clean::Vector{Bool})
                    for move in 1:moves
                        source_no = rand(1:length(sources))
                        clean[mix[:,source_no]].=false #all obs with this source in the mix matrix will be dirty
                        source_pos = rand(1:size(sources[source_no][1])[1])
                        pos_WM = exp.(sources[source_no][1][source_pos,:]) #leaving logspace

                        base_to_shift = rand(1:4)
                        permutation_sign = rand(-1:2:1)
                        shift_size = rand(PWM_shift_dist)
                        
                        for base in 1:4 #ACGT
                            if base == base_to_shift
                                pos_WM[base] =
                                clamp(0, #no lower than 0 prob
                                (pos_WM[base]          #selected PWM posn
                                + permutation_sign * shift_size), #randomly permuted by size param
                                1) #no higher than prob 1
                            else
                                size_frac = shift_size / 3 #other bases shifted in the opposite direction by 1/3 the shift accumulated at the base to permute
                                pos_WM[base] =
                                clamp(0,
                                (pos_WM[base]
                                - permutation_sign * size_frac),
                                1)
                            end
                        end
                        pos_WM = pos_WM ./ sum(pos_WM) #renormalise to sum 1 - necessary in case of clamping at 0 or 1
                        @assert HMMBase.isprobvec(pos_WM) #throw error if the position WM is invalid
                        sources[source_no][1][source_pos,:] = log.(pos_WM) # if ok reassign in logspace and carry on
                    end
                end

                function permute_source_lengths!(mix::BitMatrix, sources::Vector{Tuple{Matrix{Float64},Int64}}, priors::Vector{Vector{Dirichlet{Float64}}}, moves::Int64, length_limits::UnitRange{Int64}, clean::Vector{Bool}, uninformative::Dirichlet=Dirichlet([.25,.25,.25,.25]))
                    for move in 1:moves
                        source_no = rand(1:length(sources))
                        clean[mix[:,source_no]].=false #all obs with this source in the mix matrix will be dirty
                        source_PWM, prior_idx = sources[source_no]
                        source_length = size(source_PWM)[1]
                        prior = priors[source_no]

                        #randomly chose to extend or contract a source, but prevent an extension or contraction of a source at its length limit if the randomly chosen sign would violate it- in this case use the opposite sign
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
                            @assert HMMBase.isprobvec(exp.(new_WM)) #assert that it's a valid probability vec
                            permute_5_pr ? #if we're extending the 5' end of the matrix
                                (source_PWM = vcat(transpose(new_WM), source_PWM); #push the new WM to the front
                                prior_idx -= 1) :#and decrement the prior index 
                                source_PWM = vcat(source_PWM, transpose(new_WM)) #if we're exending the 3' end, push it to the end 
                        else #if we're to remove a position
                            permute_5_pr ? (source_PWM=source_PWM[2:end,:] ; prior_idx += 1) :
                                            source_PWM=source_PWM[1:end-1,:] #do it from the proper end, incrementing prior index for 5' subtraction
                        end
                        sources[source_no] = (source_PWM, prior_idx) #update the sources vector
                    end
                end

#iterative merging with other models in the ensemble
function merge_model!(models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, iterates::Int64)
    m.log_likelihood = -Inf; iterate = 1 #init for iterative likelihood search
    T,O = size(observations); T=T-1
    S = length(m.sources)

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mixing_matrix, true, true)
    clean=Vector{Bool}(trues(O))


    while m.log_likelihood < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        merge_model = deserialize(rand(models).path) #randomly select a model to merge
        s = rand(1:S)
        m.sources[s] = merge_model.sources[s]
        m.mixing_matrix[:,s] = merge_model.mixing_matrix[:,s]
        clean[m.mixing_matrix[:,s]].=false

        m.log_likelihood, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mixing_matrix, true, true, cache, clean)
        iterate += 1
    end
end

function merge_model!(librarian::Int64, models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, iterates::Int64)
    m.log_likelihood = -Inf; iterate = 1 #init for iterative likelihood search
    T,O = size(observations); T=T-1
    S = length(m.sources)

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mixing_matrix, true, true)
    clean=Vector{Bool}(trues(O))


    while m.log_likelihood < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        merge_model = remotecall_fetch(deserialize, librarian, rand(models).path) #randomly select a model to merge
        s = rand(1:S)
        m.sources[s] = merge_model.sources[s]
        m.mixing_matrix[:,s] = merge_model.mixing_matrix[:,s]
        clean[m.mixing_matrix[:,s]].=false

        m.log_likelihood, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mixing_matrix, true, true, cache, clean)
        iterate += 1
    end
end

#iterative source reinitialisation from priors
function reinit_sources!(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, iterates::Int64)
    m.log_likelihood = -Inf; iterate = 1 #init for iterative likelihood search
    T,O = size(observations); T=T-1
    S = length(m.sources)

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mixing_matrix, true, true)
    clean=Vector{Bool}(trues(O))

    while m.log_likelihood < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        s_to_reinit=rand(1:S) # pick a random source to reinitialise
        m.sources[s_to_reinit] = init_logPWM_sources([source_priors[s_to_reinit]], m.source_length_limits)[1]
        m.mixing_matrix[:,s_to_reinit] = init_mixing_matrix(mix_prior,O,1)
        clean[m.mixing_matrix[:,s_to_reinit]].=false

        m.log_likelihood, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mixing_matrix, true, true, cache, clean)
        iterate += 1
    end
end