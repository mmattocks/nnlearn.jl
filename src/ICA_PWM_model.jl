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
    score_mat, source_bitindex, source_wmls=score_sources(sources, observations, obs_lengths, mix)
    log_lh = IPM_likelihood(sources, obs_lengths, score_mat, source_bitindex, bg_scores, mix, source_wmls)

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
function score_sources(sources::Vector{Tuple{Matrix{Float64},Int64}}, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, mix_matrix::BitMatrix; scores::Matrix{Matrix{Float64}}=Matrix{Matrix{Float64}}(undef,(size(observations)[2],length(sources))), source_bitindex::BitArray=falses((size(observations)[1]-1), length(sources), size(observations)[2]), clean_matrix::BitMatrix=falses(size(mix_matrix)), revcomp=true) #scores:OxS matrix of score matrices; source_bitindex: TxSxO bitcube indexing which T values have scores for each SxO; clean matrix: tracks indices that are clean and do not need to be recalculated in permutations
    calc_matrix = BitMatrix(mix_matrix.-clean_matrix) #when permuting, don't want to recalculate oxs indices that were not affected by permutation
    indices     = findall(calc_matrix) #indices to calclulate
    source_wmls = [size(source[1])[1] for source in sources]

    Threads.@threads for idx in indices
        o,s     = idx[1],idx[2]
        source  = sources[s][1] #get the PWM from the source tuple
        wml     = source_wmls[s] #weight matrix length
        obsl = obs_lengths[o] #observation length
        source_stop=obsl-wml+1 #stop scannng th source across the observation here

        scores[o,s] = nnlearn.score_source(observations[1:obsl,o], source, source_stop, revcomp) #get the scores for this oxs

        source_bitindex[wml:obsl,s,o] .= true #set bitindex from the initial source emission base to the end of the observations matrix to true
    end

    return scores, source_bitindex, source_wmls
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

function IPM_likelihood(sources::Vector{Tuple{Matrix{Float64},Int64}}, obs_lengths::Vector{Int64}, score_mat::Matrix{Matrix{Float64}}, source_bitindex::BitArray, bg_scores::AbstractArray{Float64}, mix::BitMatrix, source_wmls::Vector{Int64}, revcomp::Bool=true)
    revcomp ? log_motif_expectation = log(0.5 / size(bg_scores)[1]) : log_motif_expectation = log(1 / size(bg_scores)[1])#log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise
    O = size(bg_scores)[2]
    obs_lhs=zeros(O)
    Threads.@threads for o in 1:O
        mixview=view(mix,o,:)
        obs_source_indices = findall(mixview)
        obs_source_bitindex = view(source_bitindex,:,mixview,o)
        obs_cardinality = length(obs_source_indices) #the more sources, the greater the cardinality_penalty
        obs_cardinality > 0 ? cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality)) : cardinality_penalty = 0.0
            
        obs_lhs[o] = nnlearn.weave_scores(obs_lengths[o], view(bg_scores,:,o), view(score_mat,o,:), obs_source_indices, obs_source_bitindex, view(source_wmls,:), log_motif_expectation, cardinality_penalty, revcomp)
    end

    return CLHMM.lps(obs_lhs)
end

                function weave_scores(obsl::Int64, bg_scores::SubArray, score_mat::SubArray, obs_source_indices::Vector{Int64}, obs_source_bitindex::SubArray, source_wmls::SubArray, log_motif_expectation::Float64, cardinality_penalty::Float64,  revcomp::Bool=true)
                    L=obsl+1
                    lh_vec = zeros(L)#likelihood vector is one position (0 initialiser) longer than the observation
                    emit_start_idxs=[findfirst(obs_source_bitindex[:, s]) for s in 1:length(obs_source_indices)] #construct an array of the start index for every source's contribution scores

                    for i in 2:L #i=1 is ithe lh_vec initializing 0, i=2 is the score of the first background position (ie t=1)
                        t=i-1
                        score = lh_vec[i-1] + bg_scores[t] + cardinality_penalty

                        t_sources = obs_source_indices[obs_source_bitindex[t,:]]
                        emit_starts = emit_start_idxs[obs_source_bitindex[t,:]]
                        for (n,s) in enumerate(t_sources)
                            wml = source_wmls[s]
                            from_score = lh_vec[i-wml+1] #score at the first position of the PWM
                            score_array = score_mat[s]
                            score_idx = t - emit_starts[n] + 1
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
function permute_model!(m::ICA_PWM_model, model_no::Int64, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64=10, permutation_moves::Int64=100, move_type_weights::Vector{Float64}=ones(3)/3, PWM_shift_dist::Distributions.Uniform=Uniform(.0001,.02)) #shift_dist is given in decimal probability values- converted to log space in permute_source_lengths!

    m.name = string(model_no); m.log_likelihood = -Inf; iterate = 1 #init for iterative likelihood search
    mix_moves, PWM_weight_moves, PWM_length_moves = Int.(round.(permutation_moves .* move_type_weights)) #divide total number of moves between three move types by supplied weights
    mix_moves > length(m.mixing_matrix) && (mix_moves = length(m.mixing_matrix))
    T,O = size(observations); T=T-1
    S = length(m.sources)

    score_mat, source_bitindex, source_wmls=score_sources(m.sources, observations, obs_lengths, m.mixing_matrix) #get scores and indices once before iterating
    clean = m.mixing_matrix

    while m.log_likelihood < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates

        mix_moves > 0 && mix_matrix_decorrelate!(m.mixing_matrix, mix_moves, clean)

        PWM_weight_moves > 0 && permute_source_weights!(m.sources, PWM_weight_moves, PWM_shift_dist, clean)

        PWM_length_moves > 0 && permute_source_lengths!(m.sources, priors, PWM_length_moves, m.source_length_limits, clean)

        score_mat, source_bitindex, source_wmls=score_sources(m.sources, observations, obs_lengths, m.mixing_matrix)
        m.log_likelihood = IPM_likelihood(m.sources, obs_lengths, score_mat, source_bitindex, bg_scores, m.mixing_matrix, source_wmls)
    
        iterate += 1
    end
end

                #PERMUTATION SUBFUNCS
                function mix_matrix_decorrelate!(mix::BitMatrix, moves::Int64, clean::BitMatrix)
                    indices_to_flip = Vector{CartesianIndex{2}}()
                    for i in 1:moves
                        index_to_flip = rand(CartesianIndices(mix))
                        while index_to_flip in indices_to_flip #sampling without replacement, want to always get the requested # of indices
                            index_to_flip = rand(CartesianIndices(mix))
                        end
                        push!(indices_to_flip, index_to_flip)
                    end
                    mix[indices_to_flip] .= .!mix[indices_to_flip]
                    clean[indices_to_flip] .= false #track dirty indices
                end

                function permute_source_weights!(sources::Vector{Tuple{Matrix{Float64},Int64}}, moves::Int64, PWM_shift_dist::Distributions.Uniform, clean::BitMatrix)
                    for move in 1:moves
                        source_no = rand(1:length(sources))
                        clean[:,source_no].=false #all obs with this source in the mix matrix will be dirty
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

                function permute_source_lengths!(sources::Vector{Tuple{Matrix{Float64},Int64}}, priors::Vector{Vector{Dirichlet{Float64}}}, moves::Int64, length_limits::UnitRange{Int64}, clean::BitMatrix, uninformative::Dirichlet=Dirichlet([.25,.25,.25,.25]))
                    for move in 1:moves
                        source_no = rand(1:length(sources))
                        clean[:,source_no].=false #all obs with this source in the mix matrix will be dirty
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
function merge_model!(models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, model_no::Int64, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, iterates::Int64)
    m.name = string(model_no); m.log_likelihood = -Inf; iterate = 1 #init for iterative likelihood search
    T,O = size(observations); T=T-1
    S = length(m.sources)

    score_mat, source_bitindex, source_wmls=score_sources(m.sources, observations, obs_lengths, m.mixing_matrix) #get scores and indices once before iterating
    clean = m.mixing_matrix

    while m.log_likelihood < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        merge_model = deserialize(rand(models).path) #randomly select a model to merge
        s = rand(1:S)
        m.sources[s] = merge_model.sources[s]
        m.mixing_matrix[:,s] = merge_model.mixing_matrix[:,s]
        clean[:, s] .= false

        score_mat, source_bitindex, source_wmls=score_sources(m.sources, observations, obs_lengths, m.mixing_matrix)
        m.log_likelihood = IPM_likelihood(m.sources, obs_lengths, score_mat, source_bitindex, bg_scores, m.mixing_matrix, source_wmls)
        iterate += 1
    end
end

#iterative source reinitialisation from priors
function reinit_sources!(m::ICA_PWM_model, model_no::Int64, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, iterates::Int64)
    m.name = string(model_no); m.log_likelihood = -Inf; iterate = 1 #init for iterative likelihood search
    T,O = size(observations); T=T-1
    S = length(m.sources)

    score_mat, source_bitindex, source_wmls=score_sources(m.sources, observations, obs_lengths, m.mixing_matrix) #get scores and indices once before iterating)
    clean = m.mixing_matrix

    while m.log_likelihood < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        s_to_reinit=rand(1:S) # pick a random source to reinitialise
        m.sources[s_to_reinit] = init_logPWM_sources([source_priors[s_to_reinit]], m.source_length_limits)[1]
        m.mixing_matrix[:,s_to_reinit] = init_mixing_matrix(mix_prior,O,1)
        clean[:, s_to_reinit] .= false

        score_mat, source_bitindex, source_wmls=score_sources(m.sources, observations, obs_lengths, m.mixing_matrix)
        m.log_likelihood = IPM_likelihood(m.sources, obs_lengths, score_mat, source_bitindex, bg_scores, m.mixing_matrix, source_wmls)
        iterate += 1
    end
end