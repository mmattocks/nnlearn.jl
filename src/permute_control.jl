#permutation routine function- 
#general logic: receive array of permutation parameters, until a model more likely than the least is found:
#randomly select a model from the ensemble (the least likely having been removed by this point), then sample new models by permuting with each of hte given parameter sets until a model more likely than the current contour is found
#if none is found for the candidate model, move on to another candidate until the models_to_permute iterate is reached, after which return nothing for an error code

#four permutation modes: source (iterative random changes to sources until model lh>contour or iterate limit reached)
#						-(iterates, weight shift freq per source base, length change freq per source, weight_shift_dist (a ContinuousUnivariateDistribution)) for permute params
#						mix (iterative random changes to mix matrix as above)
#						-(iterates, unitrange of # of moves)
#						init (iteratively reinitialize sources from priors)
#						-(iterates) for init params
#						merge (iteratively copy a source + mix matrix row from another model in the ensemble until lh>contour or iterate						limit reached)
#						-(iterates) for merpge params

function run_permutation_routine(e::Bayes_IPM_ensemble, job_sets::Vector{Tuple{Vector{Tuple{String,Any}},Vector{AbstractFloat}}}, job_set_thresh::Vector{AbstractFloat}, job_limit::Integer, models_to_permute::Integer, contour::AbstractFloat)
    start=time()
    job_set,job_weights=job_sets[findlast(thresh->(contour>thresh),job_set_thresh)]
    !(length(job_weights)==length(job_set)) && throw(ArgumentError("Job set and job weight vec must be same length!"))

	for model = 1:models_to_permute
		m_record = rand(e.models)
        m = deserialize(m_record.path)

        for job in 1:job_limit
            mode, params = get_job(job_set,job_weights,m.flags,m.sources,m.source_length_limits)
            if mode == "PS"
                new_m=permute_source(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
            elseif mode == "PM"
                new_m=permute_mix(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
            elseif mode == "PSFM"
				new_m=perm_src_fit_mix(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
            elseif mode == "FM"
                new_m=fit_mix(m, e.obs_array, e.obs_lengths, e.bg_scores)
            elseif mode == "DM"
                new_m=distance_merge(e.models, m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
            elseif mode == "SM"
                new_m=similarity_merge(e.models, m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
			elseif mode == "RD"
				new_m=random_decorrelate(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
			elseif mode == "RI"
				new_m=reinit_src(m, contour,  e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, e.mix_prior, params...)
            elseif mode == "EM"
                new_m=erode_model(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                occursin("PSFM",new_m.flags[1]) && (mode="PSFM")
            else
                @error "Malformed permute mode code! Current supported: \"PS\", \"PM\", \"PSFM\", \"FM\", \"DM\", \"SM\",\"RD\", \"RI\", \"EM\""
            end

			dupecheck(new_m,m) && new_m.log_Li > contour && return new_m, (time()-start, job, model, m.log_Li, new_m.log_Li, mode)
		end
	end
	return nothing, nothing
end

function worker_permute(e::Bayes_IPM_ensemble, job_chan::RemoteChannel, models_chan::RemoteChannel, job_sets::Vector{Tuple{Vector{Tuple{String,Any}},Vector{AbstractFloat}}}, job_set_thresh::Vector{AbstractFloat}, job_limit::Integer, models_to_permute::Integer)
	persist=true
    id=myid()
    model_ctr=1
	while persist
        wait(job_chan)

        start=time()

		models = fetch(job_chan)
        models === nothing && (persist=false) && break
		contour, ll_idx = findmin([model.log_Li for model in models])
		deleteat!(models, ll_idx)
        job_set,job_weights=job_sets[findlast(thresh->(contour>thresh),job_set_thresh)]
        !(length(job_weights)==length(job_set)) && throw(ArgumentError,"Job set and job weight vec must be same length!")

		for model=1:models_to_permute
			found::Bool=false
			m_record = rand(models)
			m = remotecall_fetch(deserialize,1,m_record.path)
            for job in 1:job_limit
                mode, params = get_job(job_set,job_weights,m.flags,m.sources,m.source_length_limits)
                if mode == "PS"
                    new_m=permute_source(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "PM"
                    new_m=permute_mix(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "PSFM"
                    new_m=perm_src_fit_mix(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "FM"
                    new_m=fit_mix(m, e.obs_array, e.obs_lengths, e.bg_scores)
                elseif mode == "DM"
                    new_m=distance_merge(e.models, m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "SM"
                    new_m=similarity_merge(e.models, m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "RD"
                    new_m=random_decorrelate(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                elseif mode == "RI"
                    new_m=reinit_src(m, contour,  e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, e.mix_prior, params...)
                elseif mode == "EM"
                    new_m=erode_model(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
                    occursin("PSFM",new_m.flags[1]) && (mode="PSFM")
                else
                    @error "Malformed permute mode code! Current supported: \"PS\", \"PM\", \"PSFM\", \"FM\", \"DM\", \"SM\",\"RD\", \"RI\", \"EM\""
                end
				dupecheck(new_m,m) && new_m.log_Li > contour && (put!(models_chan, (new_m ,id, (time()-start, job, model_ctr, m.log_Li, new_m.log_Li, mode))); found=true; model_ctr=1; break)
			end
            found==true && break;
            model_ctr+=1
            wait(job_chan)
            fetch(job_chan)!=models && (break) #if the ensemble has changed during the search, update it
			model==models_to_permute && (put!(models_chan,nothing);persist=false)#worker to put nothing on channel if it fails to find a model more likely than contour
		end
	end
end

                function get_job(job_set, job_weights, flags, srcs, src_lims)
                    job_dist=Categorical(job_weights)        
                    mode,params=job_set[rand(job_dist)]

                    if ("nofit" in flags)
                        while mode=="FM"
                            mode, params =job_set[rand(job_dist)]
                        end
                    end
        
                    if mode=="EM" && !any([>(size(srcs[s][1],1), src_lims[1]) for s in 1:length(srcs)]) 
                        while mode=="EM"
                            mode,params=job_set[rand(job_dist)]
                        end
                    end
                    return mode,params
                end

                function dupecheck(new_model, model)
                    (new_model.sources==model.sources && new_model.mix_matrix==model.mix_matrix) ? (return false) : (return true)
                end