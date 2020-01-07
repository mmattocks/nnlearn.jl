#DECORRELATION SEARCH PATTERNS
#random permutation of single sources until model with log likelihood>contour found or iterates limit reached. will always produce at least one weight shift per iteration for weight_shift_freq>0, more than this or length changes depend on the supplied probabilities. one length change per iterate
function permute_source(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64=length(m.sources), weight_shift_freq::Float64=.5, length_change_freq::Float64=1.0, weight_shift_dist::Distributions.ContinuousUnivariateDistribution=Weibull(1.5,.1)) 
#weight_shift_dist is given in decimal probability values- converted to log space in permute_source_lengths!
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources);
    flags=deepcopy(m.flags); flags[1]="PS from $(m.name)"

    a, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources);
        s = rand(1:S)
        clean=Vector{Bool}(trues(O))
        clean[m.mix_matrix[:,s]].=false #all obs with source are dirty

        weight_shift_freq > 0 && (new_sources[s]=permute_source_weights(new_sources[s], weight_shift_freq, weight_shift_dist))
        rand() < length_change_freq && (new_sources[s]=permute_source_length(new_sources[s], source_priors[s], m.source_length_limits))
        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true, cache, clean)
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, m.mix_matrix, new_log_Li, flags), time()-instrct_start) : (return consolidate_srcs(cons_idxs, new_sources, m.mix_matrix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits), time()-instrct_start)
end

function permute_mix(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64=10, mix_move_range::UnitRange=Int(ceil(.001*length(m.mix_matrix))):length(m.mix_matrix)) 
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_mix=falses(size(m.mix_matrix))
    flags=deepcopy(m.flags); flags[1]="PM from $(m.name)"
    "nofit" in flags && deleteat!(flags, findfirst(isequal("nofit"),flags))

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true)
    dirty=false

    while (new_log_Li <= contour || !dirty) && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        mix_moves=rand(mix_move_range)
        mix_moves > length(m.mix_matrix) && (mix_moves = length(m.mix_matrix))
    
        new_mix, clean = mix_matrix_decorrelate(m.mix_matrix, mix_moves) #generate a decorrelated candidate mix
        c_log_li, c_cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #calculate the model with the candidate mix
        positive_indices=c_cache.>(cache) #obtain any obs indices that have greater probability than we started with
        clean=Vector{Bool}(trues(O)-positive_indices)
        if any(positive_indices) #if there are any such indices
            new_log_Li, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #calculate the new model
            dirty=true
        end
        iterate += 1
    end

    return ICA_PWM_model("candidate",m.sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li, flags)
end

function perm_src_fit_mix(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64=length(m.sources), weight_shift_freq::Float64=.25, length_change_freq::Float64=.5, weight_shift_dist::Distributions.ContinuousUnivariateDistribution=Weibull(1.0,.1))
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
    flags=deepcopy(m.flags); flags[1]="PSFM from $(m.name)"

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix); tm_one=deepcopy(m.mix_matrix);
        clean=Vector{Bool}(trues(O))
        s = rand(1:S); 
        clean[new_mix[:,s]].=false #all obs starting with source are dirty

        new_mix[:,s].=false;tm_one[:,s].=true

        weight_shift_freq > 0 && (new_sources[s]=permute_source_weights(new_sources[s], weight_shift_freq, weight_shift_dist))
        rand() < length_change_freq && (new_sources[s]=permute_source_length(new_sources[s], source_priors[s], m.source_length_limits))

        l,zero_cache=IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true)
        l,one_cache=IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, tm_one, true, true)

        fit_mix=one_cache.>=zero_cache

        new_mix[:,s]=fit_mix

        clean[fit_mix].=false #all obs ending with source are dirty

        new_log_Li = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, false, zero_cache, clean)
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li, flags)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function fit_mix(m::ICA_PWM_model, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, exclude_src::Int64=0)
    instrct_start=time(); T,O = size(observations); T=T-1; S = length(m.sources)
    new_mix=deepcopy(m.mix_matrix); test_mix=falses(size(m.mix_matrix))
    flags=deepcopy(m.flags); flags[1]="FM from $(m.name)"; push!(flags,"nofit")

    l,zero_cache=IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, test_mix, true, true)

    for (s,source) in enumerate(m.sources)
        if s!=exclude_src
            test_mix=falses(size(m.mix_matrix))
            test_mix[:,s].=true

            l,src_cache=IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, test_mix, true, true)

            fit_mix=src_cache.>=zero_cache
            new_mix[:,s]=fit_mix
        end
    end

    new_mix==m.mix_matrix ? new_log_Li=-Inf : new_log_Li = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, new_mix, true)
    return ICA_PWM_model("candidate",m.sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li, flags)
end

function random_decorrelate(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64=length(m.sources), weight_shift_freq::Float64=.1, length_change_freq::Float64=.3, weight_shift_dist::Distributions.ContinuousUnivariateDistribution=Weibull(1.5,.1), mix_move_range::UnitRange=1:Int(ceil(size(m.mix_matrix,1)*.1)))
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
    flags=deepcopy(m.flags); flags[1]="RD from $(m.name)"
    "nofit" in flags && deleteat!(flags, findfirst(isequal("nofit"),flags))

    a, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))
        s = rand(1:length(m.sources))
        clean[new_mix[:,s]].=false #all obs starting with source are dirty
        weight_shift_freq > 0 && (new_sources[s]=permute_source_weights(new_sources[s], weight_shift_freq, weight_shift_dist))
        rand() < length_change_freq && (new_sources[s]=permute_source_length(new_sources[s], source_priors[s], m.source_length_limits))
        new_mix[:,s]=mixvec_decorrelate(new_mix[:,s],rand(mix_move_range))
        clean[new_mix[:,s]].=false #all obs ending with source are dirty
        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean)
        iterate += 1
    end


    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li, ["RD from $(m.name)"])) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function distance_merge(models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64)
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
    flags=deepcopy(m.flags); flags[1]="DM from $(m.name)"
    "nofit" in flags && deleteat!(flags, findfirst(isequal("nofit"), flags))

    a, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))

        merger_m = deserialize(rand(models).path) #randomly select a model to merge
        s = rand(1:S) #randomly select a source to merge
        s > m.informed_sources ? #if the source is on an uninformative mix prior, the merger model source will be selected by mixvector similarity
            merge_s=most_dissimilar(new_mix,merger_m.mix_matrix) : merge_s=s
        
        clean[new_mix[:,s]].=false #mark dirty any obs that start with the source
        new_sources[s] = merger_m.sources[merge_s] #copy the source
        new_mix[:,s] = merger_m.mix_matrix[:,merge_s]
        clean[new_mix[:,s]].=false #mark dirty any obs that end with the source

        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li,flags)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function distance_merge(models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64)
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
    flags=deepcopy(m.flags); flags[1]="DM from $(m.name)"
    "nofit" in flags && deleteat!(flags, findfirst(isequal("nofit"),flags))

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))
        merger_m = remotecall_fetch(deserialize, 1, rand(models).path) #randomly select a model to merge
        s = rand(1:S) #randomly select a source to merge
        s > m.informed_sources ? #if the source is on an uninformative prior, the merger model source will be selected by mixvector dissimilarity
            merge_s=most_dissimilar(new_mix,merger_m.mix_matrix) : merge_s=s
        
        clean[new_mix[:,s]].=false #mark dirty any obs that start with the source
        new_sources[s] = merger_m.sources[merge_s] #copy the source
        new_mix[:,s] = merger_m.mix_matrix[:,merge_s]
        clean[new_mix[:,s]].=false #mark dirty any obs that end with the source

        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li,flags)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function similarity_merge(models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64)
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources)
    flags=deepcopy(m.flags); flags[1]="SM from $(m.name)"
    "nofit" in flags && deleteat!(flags, findfirst(isequal("nofit"),flags))

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))

        merger_m = deserialize(rand(models).path) #randomly select a model to merge
        s = rand(1:S) #randomly select a source in the model to merge
        s > m.informed_sources ? #if the source is on an uninformative prior, the merger model source will be selected by mixvector similarity
        merge_s=most_similar(m.mix_matrix[:,s],merger_m.mix_matrix) : merge_s=s

        clean[m.mix_matrix[:,s]].=false #mark dirty any obs that have the source
        new_sources[s] = merger_m.sources[merge_s] #copy the source

        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, m.mix_matrix, new_log_Li,flags)) : (return consolidate_srcs(cons_idxs, new_sources, m.mix_matrix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function similarity_merge(models::Vector{nnlearn.Model_Record}, m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64)
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources)
    flags=deepcopy(m.flags); flags[1]="SM from $(m.name)"
    "nofit" in flags && deleteat!(flags, findfirst(isequal("nofit"),flags))

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true)
    clean=Vector{Bool}(trues(O))

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))

        merger_m = remotecall_fetch(deserialize, 1, rand(models).path) #randomly select a model to merge
        s = rand(1:S) #randomly select a source in the model to merge
        s > m.informed_sources ? #if the source is on an uninformative prior, the merger model source will be selected by mixvector similarity
        merge_s=most_similar(m.mix_matrix[:,s],merger_m.mix_matrix) : merge_s=s

        clean[m.mix_matrix[:,s]].=false #mark dirty any obs that have the source
        new_sources[s] = merger_m.sources[merge_s] #copy the source

        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true, cache, clean) #assess likelihood

        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, m.mix_matrix, new_log_Li,flags)) : (return consolidate_srcs(cons_idxs, new_sources, m.mix_matrix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function reinit_src(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, iterates::Int64, inform_mix_prior::Bool=true)
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
    flags=deepcopy(m.flags); flags[1]="RS from $(m.name)"

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix); tm_one=deepcopy(m.mix_matrix);
        clean=Vector{Bool}(trues(O))
        s = rand(1:S); 
        clean[new_mix[:,s]].=false #all obs starting with source are dirty

        new_mix[:,s].=false;tm_one[:,s].=true

        new_sources[s] = init_logPWM_sources([source_priors[s]], m.source_length_limits)[1] #reinitialize the source

        l,zero_cache=IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, true)
        l,one_cache=IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, tm_one, true, true)

        fit_mix=one_cache.>=zero_cache

        new_mix[:,s]=fit_mix

        clean[fit_mix].=false #all obs ending with source are dirty

        new_log_Li = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, new_mix, true, false, zero_cache, clean)
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, new_mix, new_log_Li,flags)) : (return consolidate_srcs(cons_idxs, new_sources, new_mix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

function erode_model(m::ICA_PWM_model, contour::Float64, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, iterates::Int64, info_thresh::Float64=.25)
    instrct_start=time(); new_log_Li=-Inf;  iterate = 1
    T,O = size(observations); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources)
    flags=deepcopy(m.flags); flags[1]="EM from $(m.name)"

    erosion_sources=Vector{Int64}()
    for (s,src) in enumerate(m.sources)
        pwm,pi=src
        if size(pwm,1)>m.source_length_limits[1]
            infovec=get_pwm_info(pwm)
            any(info->>(info, info_thresh),infovec) && push!(erosion_sources,s)
        end
    end

    if length(erosion_sources)==0 #if we got a model we cant erode bail out to PSFM
        return perm_src_fit_mix(m, contour, observations,obs_lengths,bg_scores,source_priors, iterates)
    end

    a, cache = IPM_likelihood(m.sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        clean=Vector{Bool}(trues(O))

        s=rand(erosion_sources)

        new_sources[s]=erode_source(new_sources[s], m.source_length_limits)
        clean[m.mix_matrix[:,s]].=false

        new_log_Li, cache = IPM_likelihood(new_sources, observations, obs_lengths, bg_scores, m.mix_matrix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_model("candidate",new_sources, m.informed_sources, m.source_length_limits, m.mix_matrix, new_log_Li,flags)) : (return consolidate_srcs(cons_idxs, new_sources, m.mix_matrix, observations, obs_lengths, bg_scores, source_priors, m.informed_sources, m.source_length_limits))
end

##ORTHOGONALITY HELPER
function consolidate_srcs(con_idxs::Vector{Vector{Int64}}, sources::Vector{Tuple{Matrix{Float64},Int64}}, mix::BitMatrix, observations::Matrix{Int64}, obs_lengths::Vector{Int64}, bg_scores::Matrix{Float64}, source_priors::Vector{Vector{Dirichlet{Float64}}}, informed_sources::Int64, source_length_limits::UnitRange)
    new_sources=deepcopy(sources);new_mix=deepcopy(mix)
    consrc=0
    for (s,convec) in enumerate(con_idxs)
        if length(convec)>0
            consrc=s
            for cons in convec
                new_sources[cons]=init_logPWM_sources([source_priors[cons]], source_length_limits)[1]
                new_mix[:,s]=[any([new_mix[i,s],mix[i,cons]]) for i in 1:size(mix,1)] #consolidate the new_mix source with the consolidated mix
            end
            break #dont do this for more than one consolidation at a time
        end
    end

    return fit_mix(ICA_PWM_model("consolidate", new_sources, informed_sources, source_length_limits, new_mix, -Inf, [""]), observations, obs_lengths, bg_scores, consrc)
end

function consolidate_check(sources::Vector{Tuple{Matrix{Float64},Int64}}; thresh=.035)
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
#SOURCE PERMUTATION
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
                    !isprobvec(new_wm) && throw(DomainError("Bad weight vector generated in wm_shift! $new_wm")) #throw assertion exception if the position WM is invalid
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
                !isprobvec(exp.(ins_WM[pos,:])) && throw(DomainError("Bad weight vector generated in permute_source_length! $new_wm"))
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

function erode_source(source::Tuple{Matrix{Float64},Int64},length_limits::UnitRange{Int64},info_thresh::Float64=.25)
    pwm,prior_idx=source
    infovec=get_pwm_info(pwm)
    start_idx,end_idx=get_erosion_idxs(infovec, info_thresh, length_limits)

    return new_source=(pwm[start_idx:end_idx,:], prior_idx+start_idx-1)
end

    function get_pwm_info(pwm::Matrix{Float64}; logsw::Bool=true)
        wml=size(pwm,1)
        infovec=zeros(wml)
        for pos in 1:wml
            logsw ? wvec=deepcopy(exp.(pwm[pos,:])) : wvec=deepcopy(pwm[pos,:])
            !isprobvec(wvec) && throw(DomainError("Bad wvec in get_pwm_info $wvec -Original sources must be in logspace!!"))
            wvec.+=10^-99
            infscore = (2.0 + sum([x*log(2,x) for x in wvec]))
            infovec[pos]=infscore
        end
        return infovec
    end

    function get_erosion_idxs(infovec::Vector{Float64}, info_thresh::Float64, length_limits::UnitRange{Int64})
        srcl=length(infovec)
        contractable =  srcl-length_limits[1]
        contractable <=0 && throw(DomainError("erode_source passed a source at its lower length limit!"))
        centeridx=findmax(infovec)[2]
        
        start_idx=findprev(info-><(info,info_thresh),infovec,centeridx)
        start_idx===nothing ? (start_idx=1) : (start_idx+=1)
        end_idx=findnext(info-><(info, info_thresh),infovec,centeridx)
        end_idx===nothing ? (end_idx=srcl) : (end_idx-=1)

        pos_to_erode=srcl-(end_idx-start_idx)
        if pos_to_erode > contractable
            pos_to_restore = pos_to_erode-contractable
            while pos_to_restore>0
                end_die=rand()
                if end_die <= .5
                    start_idx>1 && (pos_to_restore-=1; start_idx-=1)
                else
                    end_idx<srcl && (pos_to_restore-=1; end_idx+=1)
                end
            end
        end

        return start_idx, end_idx
    end

#MIX MATRIX FUNCTIONS
function mixvec_decorrelate(mix::BitVector, moves::Int64)
    new_mix=deepcopy(mix)
    idxs_to_flip=rand(1:length(mix), moves)
    new_mix[idxs_to_flip] .= .!mix[idxs_to_flip]
    return new_mix
end

function mix_matrix_decorrelate(mix::BitMatrix, moves::Int64)
    clean=Vector{Bool}(trues(size(mix,1)))
    new_mix=deepcopy(mix)
    indices_to_flip = rand(CartesianIndices(mix), moves)
    new_mix[indices_to_flip] .= .!mix[indices_to_flip]
    clean[unique([idx[1] for idx in indices_to_flip])] .= false #mark all obs that had flipped indices dirty
    return new_mix, clean
end


function most_dissimilar(mix1, mix2)
    S1=size(mix1,2);S2=size(mix2,2)
    dist_mat=zeros(S1,S2)
    for s1 in 1:S1, s2 in 1:S2
        dist_mat[s1,s2]=sum(mix1[:,s1].==mix2[:,s2])
    end
    scores=vec(sum(dist_mat,dims=1))
    return findmin(scores)[2]
end

function most_similar(src_mixvec, target_mixmat)
    src_sim = [sum(src_mixvec.==target_mixmat[:,s]) for s in 1:size(target_mixmat,2)] #compose array of elementwise equality comparisons between mixvectors and sum to score
    merge_s=findmax(src_sim)[2] #source from merger model will be the one with the highest equality comparison score
end