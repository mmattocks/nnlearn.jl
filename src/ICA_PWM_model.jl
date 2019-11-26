struct ICA_PWM_model
    name::String #designator for saving model to posterior
    sources::Vector{Tuple{Matrix{Float64},Int64}} #vector of PWM signal sources (LOG PROBABILITY!!!) tupled with an index denoting the position of the first PWM base on the prior matrix- allows us to permute length and redraw from the appropriate prior position
    informed_sources::Int64 #number of sources with informative priors- these are not subject to frequency sorting in model mergers
    source_length_limits::UnitRange{Int64} #min/max source lengths for init and permutation
    mix_matrix::BitMatrix # obs x sources bool matrix
    log_Li::Float64
end

#ICA_PWM_model FUNCTIONS
ICA_PWM_model(name::String, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, source_length_limits::UnitRange{Int64}) = init_IPM(name, source_priors,mix_prior,bg_scores,observations,source_length_limits)

#MODEL INIT
function init_IPM(name::String, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::AbstractArray{Float64}, observations::AbstractArray{Int64}, source_length_limits::UnitRange{Int64})
    T,O = size(observations)
    S=length(source_priors)
    obs_lengths=[findfirst(iszero,observations[:,o])-1 for o in 1:size(observations)[2]]
    sources=init_logPWM_sources(source_priors, source_length_limits)
    mix=init_mix_matrix(mix_prior,O,S)
    log_lh = IPM_likelihood(sources, observations, obs_lengths, bg_scores, mix)

   return ICA_PWM_model(name, sources, size(mix_prior[1],2), source_length_limits, mix, log_lh)
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

                function init_mix_matrix(mix_prior::Tuple{BitMatrix,Float64}, no_observations::Int64, no_sources::Int64)
                    inform,uninform=mix_prior
                    if size(inform,2) > 0
                        @assert size(inform,1)==no_observations && size(inform,2)<=no_sources "Bad informative mix prior dimensions!"
                    end
                    @assert 0.0 <= uninform <=1.0 "Uninformative mix prior not between 0.0 and 1.0!"
                    mix_matrix = falses(no_observations, no_sources)
                    if size(inform,2)>0
                        mix_matrix[:,1:size(inform,2)]=inform
                    end
                    for index in CartesianIndices((1:no_observations,size(inform,2)+1:no_sources))
                        rand() <= uninform && (mix_matrix[index] = true)
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
                revcomp ? log_motif_expectation = log(2/obsl) : log_motif_expectation = log(4/obsl)#log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise
                mixview=view(mix,o,:)
                mixwmls=source_wmls[mixview]
                score_mat=score_obs_sources(sources[mixview], observations[1:obsl,o], obsl, mixwmls, revcomp=revcomp)
                obs_source_indices = findall(mixview)
                obs_cardinality = length(obs_source_indices) #the more sources, the greater the cardinality_penalty
                if obs_cardinality > 0 
                    penalty_sum = sum(exp.(fill(log_motif_expectation,obs_cardinality)))
                    penalty_sum > 1. && (penalty_sum=1.)
                    cardinality_penalty=log(1.0-penalty_sum)
                else
                    cardinality_penalty=0.0
                end

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
                        score = CLHMM.lps(lh_vec[i-1], bg_scores[t], cardinality_penalty)
                
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
