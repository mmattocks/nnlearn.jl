#permutation routine function- 
#general logic: receive array of permutation parameters, until a model more likely than the least is found:
#randomly select a model from the ensemble (the least likely having been removed by this point), then sample new models by permuting with each of hte given parameter sets until a model more likely than the current contour is found
#if none is found for the candidate model, move on to another candidate until the permute_limit iterate is reached, after which return nothing for an error code

#four permutation modes: source (iterative random changes to sources until model lh>contour or iterate limit reached)
#						-(iterates, weight shift freq per source base, length change freq per source, weight_shift_dist (a ContinuousUnivariateDistribution)) for permute params
#						mix (iterative random changes to mix matrix as above)
#						-(iterates, unitrange of # of moves)
#						init (iteratively reinitialize sources from priors)
#						-(iterates) for init params
#						merge (iteratively copy a source + mix matrix row from another model in the ensemble until lh>contour or iterate						limit reached)
#						-(iterates) for merpge params
function run_permutation_routine(e::Bayes_IPM_ensemble, param_set::Vector{Tuple{String,Any}}, permute_limit::Int64, contour::Float64; instruction_rand=true)
	start=time()
	for i = 1:permute_limit
		m_record = rand(e.models)
		m = deserialize(m_record.path)
		instruction_rand && shuffle!(param_set)
		for (job, (mode, params)) in enumerate(param_set)
			if mode == "PSFM"
				new_m=perm_src_fit_mix(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
			elseif mode == "FM"
                new_m=fit_mix(m, e.obs_array, e.obs_lengths, e.bg_scores)
            elseif mode == "merge"
                new_m=distance_merge(e.models, m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
			elseif mode == "random"
				new_m=random_decorrelate(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
			elseif mode == "reinit"
				new_m=reinit_src(m, contour,  e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, e.mix_prior, params...)
			else
				@error "Malformed permute mode code! Current supported: \"PSFM\", \"FM\", \"merge\", \"random\", \"reinit\""
            end
			new_m.log_Li > contour && return new_m, (time()-start, job, i, m.log_Li, new_m.log_Li, mode)
		end
	end
	return nothing, nothing
end

function worker_permute(e::Bayes_IPM_ensemble, librarian::Int64, job_chan::RemoteChannel, models_chan::RemoteChannel, param_set::Vector{Tuple{String,Any}}, permute_limit::Int64; instruction_rand=false, reset=true)
	persist=true
	id=myid()
	while persist
		wait(job_chan)
		models = fetch(job_chan)
		models === nothing && (persist=false) && break
		contour, ll_idx = findmin([model.log_Li for model in models])
		deleteat!(models, ll_idx)

		start=time()

		for i=1:permute_limit
			found::Bool=false
			m_record = rand(models)
			job_model = remotecall_fetch(deserialize,librarian,m_record.path)
			instruction_rand && shuffle!(param_set)

			for (job, (mode, params)) in enumerate(param_set)
                if mode == "PSFM"
                    new_m=perm_src_fit_mix(job_model, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "FM"
                    new_m=fit_mix(job_model, e.obs_array, e.obs_lengths, e.bg_scores)
                elseif mode == "merge"
                    new_m=distance_merge(librarian, e.models, job_model, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "random"
                    new_m=random_decorrelate(job_model, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "reinit"
                    new_m=reinit_src(job_model, contour,  e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, e.mix_prior, params...)
                else
                    @error "Malformed permute mode code! Current supported: \"PSFM\", \"FM\", \"merge\", \"random\", \"reinit\""
                end
				new_m.log_Li > contour && (put!(models_chan, (job_model,id, (time()-start, job, i, job_model.log_Li, new_m.log_Li, mode))); found=true; break)
				wait(job_chan)
				fetch(job_chan)!=models && (break; break) #if the ensemble has changed during the search, update it
			end
			found==true && break;
			i==permute_limit && (put!(models_chan,nothing);persist=false)#worker to put nothing on channel if it fails to find a model more likely than contour
		end
	end
end

#DECORRELATION SEARCH PATTERNS
function perm_src_fit_mix(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64=length(m.sources), weight_shift_freq::Float64=.1, length_change_freq::Float64=.3, weight_shift_dist::Distributions.ContinuousUnivariateDistribution=Weibull(1.5,.1))
    new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)

    while new_log_Li < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix); tm_one=deepcopy(m.mix_matrix);

        s = rand(1:S); 
        new_mix[:,s].=false;tm_one[:,s].=true

        weight_shift_freq > 0 && (new_sources[s]=permute_source_weights(new_sources[s], weight_shift_freq, weight_shift_dist))
        rand() < length_change_freq && (new_sources[s]=permute_source_length(new_sources[s], source_priors[s], m.source_length_limits))

        l,zero_cache=IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true)
        l,one_cache=IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, tm_one, true, true)

        fit_mix=one_cache.>=zero_cache
        new_mix[:,s]=fit_mix

        clean=Vector{Bool}(trues(O))
        clean[fit_mix].=false

        new_log_Li = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, false, zero_cache, clean)
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function fit_mix(m::ICA_PWM_model, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64})
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_mix=falses(size(m.mix_matrix)); test_mix=falses(size(m.mix_matrix))

    l,zero_cache=IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, new_mix, true, true)

    for (s,source) in enumerate(m.sources)
        test_mix=falses(size(m.mix_matrix))
        test_mix[:,s].=true

        l,src_cache=IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, test_mix, true, true)

        fit_mix=src_cache.>=zero_cache
        new_mix[:,s]=fit_mix
    end

    new_log_Li = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, new_mix, true)
    return ICA_PWM_model("candidate",m.sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li)
end

function random_decorrelate(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64=length(m.sources), weight_shift_freq::Float64=.1, length_change_freq::Float64=.3, weight_shift_dist::Distributions.ContinuousUnivariateDistribution=Weibull(1.5,.1), mix_move_range::UnitRange=1:size(m.mix_matrix,1))
    new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)

    a, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true)
    clean=Vector{Bool}(trues(O))

    while new_log_Li < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        s = rand(1:length(m.sources))
        weight_shift_freq > 0 && (new_sources[s]=permute_source_weights(new_sources[s], weight_shift_freq, weight_shift_dist))
        rand() < length_change_freq && (new_sources[s]=permute_source_length(new_sources[s], source_priors[s], m.source_length_limits))
        new_mix[:,s]=mixvec_decorrelate(new_mix[:,s],rand(mix_move_range))
        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean)
        iterate += 1
    end


    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function distance_merge(models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)

    a, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true)
    clean=Vector{Bool}(trues(O))

    while new_log_Li < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        merger_m = deserialize(rand(models).path) #randomly select a model to merge
        s = rand(1:S) #randomly select a source to merge
        s > m.informed_sources ? #if the source is on an uninformative prior, the merger model source will be selected by mixvector similarity
            merge_s=most_dissimilar(new_mix,merger_m.mix_matrix) : merge_s=s
        
        clean[new_mix[:,s]].=false #mark dirty any obs that start with the source
        new_sources[s] = merger_m.sources[merge_s] #copy the source
        new_mix[:,s] = merger_m.mix_matrix[:,merge_s]
        clean[new_mix[:,s]].=false #mark dirty any obs that end with the source

        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function distance_merge(librarian::Int64, models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true)
    clean=Vector{Bool}(trues(O))

    while new_log_Li < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        merger_m = remotecall_fetch(deserialize, librarian, rand(models).path) #randomly select a model to merge
        s = rand(1:S) #randomly select a source to merge
        s > m.informed_sources ? #if the source is on an uninformative prior, the merger model source will be selected by mixvector similarity
            merge_s=most_dissimilar(new_mix,merger_m.mix_matrix) : merge_s=s
        
        clean[new_mix[:,s]].=false #mark dirty any obs that start with the source
        new_sources[s] = merger_m.sources[merge_s] #copy the source
        new_mix[:,s] = merger_m.mix_matrix[:,merge_s]
        clean[new_mix[:,s]].=false #mark dirty any obs that end with the source

        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function reinit_src(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, iterates::Int64)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true)
    clean=Vector{Bool}(trues(O))

    while new_log_Li < contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        s_to_reinit=rand(1:S) # pick a random source to reinitialise
        clean[new_mix[:,s_to_reinit]].=false #mark dirty any obs that have the source before the renitialization

        new_sources[s_to_reinit] = init_logPWM_sources([source_priors[s_to_reinit]], m.source_length_limits)[1] #reinitialize the source

        s_to_reinit<=m.informed_sources ? (new_mix[:,s_to_reinit] = mix_prior[1][:,s_to_reinit]) : #if the source has an informative prior, assign that
            new_mix[:,s_to_reinit] = init_mix_matrix((falses(0,0),mix_prior[2]),O,1) #otherwise initialize the source's mix vector from the uninformative prior
        clean[new_mix[:,s_to_reinit]].=false #mark dirty any obs that have the source after the reinitialization

        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

##ORTHOGONALITY HELPER
function consolidate_srcs(con_idxs, sources, mix, observations, obs_lengths, bg_scores, source_priors, informed_sources, source_length_limits)
    new_sources=deepcopy(sources);new_mix=deepcopy(mix)
    for (s,convec) in enumerate(con_idxs)
        if length(convec)>0
            for cons in convec
                new_sources[cons]=init_logPWM_sources([source_priors[cons]], source_length_limits)[1]
            end
            break #dont do this for more than one consolidation at a time
        end
    end

    return fit_mix(ICA_PWM_model("candidate", new_sources, informed_sources, source_length_limits, new_mix, -Inf), observations, obs_lengths, bg_scores)
end

function consolidate_check(sources; thresh=.035)
    pass=true
    con_idxs=Vector{Vector{Int64}}()
    for (s1,src1) in enumerate(sources)
        s1_idxs=Vector{Int64}()
        for (s2,src2) in enumerate(sources)
            if !(s1==s2)
                pwm1=src1[1]; pwm2=src2[1]
                if -3<=(size(pwm1,1)-size(pwm2,1))<=3 
                    if pwm_distance(pwm1,pwm2)<thresh
                        push!(s1_idxs,s2)
                        pass=false
                    end
                end
            end
        end
        push!(con_idxs,s1_idxs)
    end
    return pass, con_idxs
end

                function pwm_distance(pwm1,pwm2)
                    minwml=min(size(pwm1,1),size(pwm2,1))
                    return sum([euclidean(exp.(pwm1[pos,:]), exp.(pwm2[pos,:])) for pos in 1:minwml])/minwml
                end


##BASIC UTILITY FUNCTIONS
function permute_source_weights(source::Tuple{Matrix{Float64},Int64}, shift_freq::Float64, PWM_shift_dist::Distributions.ContinuousUnivariateDistribution)
    dirty=false; source_length=size(source[1],1)
    new_source=deepcopy(source)

    for source_pos in 1:source_length
        if rand() <= shift_freq
            pos_WM = exp.(source[1][source_pos,:]) #leaving logspace, get the wm at that position
            new_source[1][source_pos,:] = log.(wm_shift(pos_WM, PWM_shift_dist)) #accumulate probabilty at a randomly selected base, reassign in logspace and carry on
            !dirty && (dirty=true)
        end
    end

    if !dirty #if no positions were shifted, pick one and shift
        rand_pos=rand(1:source_length)
        pos_WM = exp.(source[1][rand_pos,:])
        new_source[1][rand_pos,:]=log.(wm_shift(pos_WM, PWM_shift_dist))
    end

    return new_source
end

                function wm_shift(pos_WM::Vector{Float64}, PWM_shift_dist::Distributions.ContinuousUnivariateDistribution)
                    base_to_shift = rand(1:4) #pick a base to accumulate probability
                    permute_sign = rand(-1:2:1)
                    shift_size = rand(PWM_shift_dist)
                    new_wm=zeros(4)
                    
                    for base in 1:4 #ACGT
                        if base == base_to_shift
                            new_wm[base] =
                            clamp(0, #no lower than 0 prob
                            (pos_WM[base]          #selected PWM posn
                            + permute_sign * shift_size), #randomly permuted by size param
                            1) #no higher than prob 1
                        else
                            size_frac = shift_size / 3 #other bases shifted in the opposite direction by 1/3 the shift accumulated at the base to permute
                            new_wm[base] =
                            clamp(0,
                            (pos_WM[base]
                            - permute_sign * size_frac),
                            1)
                        end
                    end
                    new_wm = new_wm ./ sum(new_wm) #renormalise to sum 1 - necessary in case of clamping at 0 or 1
                    !HMMBase.isprobvec(new_wm) && throw(DomainError,"Bad weight vector generated in wm_shift! $new_wm") #throw assertion exception if the position WM is invalid
                    return new_wm
                end



function permute_source_length(source::Tuple{Matrix{Float64},Int64}, prior::Vector{Dirichlet{Float64}}, length_limits::UnitRange{Int64}, permute_range::UnitRange{Int64}=1:3, uninformative::Dirichlet=Dirichlet([.25,.25,.25,.25]))
    source_PWM, prior_idx = source
    source_length = size(source_PWM,1)

    permute_sign, permute_length = get_length_params(source_length, length_limits, permute_range)

    permute_sign==1 ? permute_pos = rand(1:source_length+1) :
        permute_pos=rand(1:source_length-permute_length)
    
    if permute_sign == 1 #if we're to add positions to the PWM
        ins_WM=zeros(permute_length,4)
        for pos in 1:permute_length
            prior_position=permute_pos+prior_idx
            prior_position<1 || prior_position>length(prior) ? 
                ins_WM[pos,:] = log.(transpose(rand(uninformative))) :
                ins_WM[pos,:] = log.(transpose(rand(prior[prior_position])))
                !HMMBase.isprobvec(exp.(ins_WM[pos,:])) && throw(DomainError,"Bad weight vector generated in permute_source_length! new_wm")
        end
        upstream_source=source_PWM[1:permute_pos-1,:]
        downstream_source=source_PWM[permute_pos:end,:]
        source_PWM=vcat(upstream_source,ins_WM,downstream_source)
        permute_pos==1 && (prior_idx-=permute_length)
    else #if we're to remove positions
        upstream_source=source_PWM[1:permute_pos-1,:]
        downstream_source=source_PWM[permute_pos+permute_length:end,:]
        source_PWM=vcat(upstream_source,downstream_source)
        permute_pos==1 && (prior_idx+=permute_length)
    end

    return (source_PWM, prior_idx) #return a new source
end

                function get_length_params(source_length::Int64, length_limits::UnitRange{Int64}, permute_range::UnitRange{Int64})
                    extendable = length_limits[end]-source_length
                    contractable =  source_length-length_limits[1]

                    if extendable == 0 && contractable > 0
                        permute_sign=-1
                    elseif contractable == 0 && extendable > 0
                        permute_sign=1
                    else
                        permute_sign = rand(-1:2:1)
                    end

                    permute_sign==1 && extendable<permute_range[end] && (permute_range=permute_range[1]:extendable)
                    permute_sign==-1 && contractable<permute_range[end] && (permute_range=permute_range[1]:contractable)
                    permute_length = rand(permute_range)

                    return permute_sign, permute_length
                end

function mixvec_decorrelate(mix::BitVector, moves::Int64)
    new_mix=deepcopy(mix)
    idxs_to_flip=rand(1:length(mix), moves)
    new_mix[idxs_to_flip] .= .!mix[idxs_to_flip]
    return new_mix
end

function most_dissimilar(mix1, mix2)
    S1=size(mix1,2);S2=size(mix2,2)
    dist_mat=zeros(S1,S2)
    for s1 in 1:S1, s2 in 1:S2
        sum(mix1[:,s1].==mix2[:,s2])
    end
    scores=sum(dist_mat,dims=1)
    return findmin(scores)[2]
end
