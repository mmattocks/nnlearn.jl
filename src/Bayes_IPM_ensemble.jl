mutable struct Bayes_IPM_ensemble
	ensemble_directory::String #ensemble models and popped-out posterior samples serialised here
	models::Vector{Model_Record} #ensemble keeps paths to serialised models and their likelihood tuples rather than keeping the models in memory

	log_Li::Vector{Float64} #likelihood of lowest-ranked model at iterate i
	log_Xi::Vector{Float64} #amt of prior mass included in ensemble contour at Li
	log_wi::Vector{Float64} #width of prior mass covered in this iterate
	log_Liwi::Vector{Float64} #evidentiary weight of the iterate
	log_Zi::Vector{Float64} #ensemble evidence
	Hi::Vector{Float64} #ensemble information

	obs_array::Matrix{Int64} #observations
	obs_lengths::Vector{Int64}

	source_priors::Vector{Vector{Dirichlet{Float64}}} #source pwm priors
	mix_prior::Tuple{BitMatrix,Float64} #prior on %age of observations that any given source contributes to

	bg_scores::Matrix{Float64} #precalculated background HMM scores

	sample_posterior::Bool
	retained_posterior_samples::Vector{Model_Record} #list of posterior sample records

	model_counter::Int64

	naive_lh::Float64 #the likelihood of the background model without any sources
end

####Bayes_IPM_ensemble FUNCTIONS
Bayes_IPM_ensemble(ensemble_directory::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::Matrix{Float64}, obs::Array{Int64}, source_length_limits; posterior_switch::Bool=true) =
Bayes_IPM_ensemble(
	ensemble_directory,
	assemble_IPMs(ensemble_directory, no_models, source_priors, mix_prior, bg_scores, obs, source_length_limits),
	[-Inf], #L0 = 0
	[0], #ie exp(0) = all of the prior is covered
	[-Inf], #w0 = 0
	[-Inf], #Liwi0 = 0
	[-1e300], #Z0 = 0
	[0], #H0 = 0,
	obs,
	[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]],
	source_priors,
	mix_prior,
	bg_scores, #precalculated background score
	posterior_switch,
	Vector{String}(),
	no_models+1,
	IPM_likelihood(init_logPWM_sources(source_priors, source_length_limits), obs, [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]], bg_scores, falses(size(obs)[2],length(source_priors))))

Bayes_IPM_ensemble(worker_pool::Vector{Int64}, ensemble_directory::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::Matrix{Float64}, obs::Array{Int64}, source_length_limits; posterior_switch::Bool=true) =
Bayes_IPM_ensemble(
	ensemble_directory,
	distributed_IPM_assembly(worker_pool, ensemble_directory, no_models, source_priors, mix_prior, bg_scores, obs, source_length_limits),
	[-Inf], #L0 = 0
	[0], #ie exp(0) = all of the prior is covered
	[-Inf], #w0 = 0
	[-Inf], #Liwi0 = 0
	[-1e300], #Z0 = 0
	[0], #H0 = 0,
	obs,
	[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]],
	source_priors,
	mix_prior,
	bg_scores, #precalculated background score
	posterior_switch,
	Vector{String}(),
	no_models+1,
	IPM_likelihood(init_logPWM_sources(source_priors, source_length_limits), obs, [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]], bg_scores, falses(size(obs)[2],length(source_priors))))

function assemble_IPMs(ensemble_directory::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::AbstractArray{Float64}, obs::AbstractArray{Int64}, source_length_limits::UnitRange{Int64})
	ensemble_records = Vector{Model_Record}()
	!isdir(ensemble_directory) && mkpath(ensemble_directory)

	@assert size(obs)[2]==size(bg_scores)[2]

	@showprogress 1 "Assembling IPM ensemble..." for model_no in 1:no_models
		model_path = string(ensemble_directory,'/',model_no)
		if !isfile(model_path)
			model = ICA_PWM_model(string(model_no), source_priors, mix_prior, bg_scores, obs, source_length_limits)
			serialize(model_path, model) #save the model to the ensemble directory
			push!(ensemble_records, Model_Record(model_path,model.log_likelihood))
		else #interrupted assembly pick up from where we left off
			model = deserialize(model_path)
			push!(ensemble_records, Model_Record(model_path,model.log_likelihood))
		end
	end

	return ensemble_records
end

function distributed_IPM_assembly(worker_pool::Vector{Int64}, ensemble_directory::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::AbstractArray{Float64}, obs::AbstractArray{Int64}, source_length_limits::UnitRange{Int64})
	ensemble_records = Vector{Model_Record}()
	!isdir(ensemble_directory) && mkpath(ensemble_directory)

	@assert size(obs)[2]==size(bg_scores)[2]

    model_chan= RemoteChannel(()->Channel{ICA_PWM_model}(length(worker_pool)))
    job_chan = RemoteChannel(()->Channel{Union{Tuple,Nothing}}(1))
	put!(job_chan,(source_priors, mix_prior, bg_scores, obs, source_length_limits))
	
    for worker in worker_pool
        remote_do(worker_assemble, worker, job_chan, model_chan)
	end
	
	assembly_progress=Progress(no_models, desc="Assembling IPM ensemble...")

	model_counter=check_assembly!(ensemble_records, ensemble_directory, no_models, assembly_progress)

	while model_counter <= no_models
		wait(model_chan)
		model=take!(model_chan)
		model.name=string(model_counter)
		model_path=string(ensemble_directory,'/',model_counter)
		serialize(model_path,model)
		push!(ensemble_records, Model_Record(model_path,model.log_likelihood))
		model_counter+=1
		next!(assembly_progress)
	end

	take!(job_chan),put!(job_chan,nothing)

	return ensemble_records
end
				function check_assembly!(ensemble_records::Vector{Model_Record}, ensemble_directory::String, no_models::Int64, assembly_progress::Progress)
					counter=1
					while counter <= no_models
						model_path=string(ensemble_directory,'/',counter)
						if isfile(model_path)
							model=deserialize(model_path)
							push!(ensemble_records, Model_Record(model_path,model.log_likelihood))
							counter+=1
							next!(assembly_progress)
						else
							return counter
						end
					end
					return counter
				end

				function worker_assemble(job_chan::RemoteChannel, models_chan::RemoteChannel) #NEEDS REWRITING, TOO SLOW IN DISTRIBUTED
					wait(job_chan)
					params=fetch(job_chan)
					while isready(job_chan)
						model=ICA_PWM_model(string(myid()),params...)
						put!(models_chan,model)
					end
				end


#permutation routine function- 
#general logic: receive array of permutation parameters, until a model more likely than the least is found:
#randomly select a model from the ensemble (the least likely having been removed by this point), then sample new models by permuting with each of hte given parameter sets until a model more likely than the current contour is found
#if none is found for the candidate model, move on to another candidate until the models_to_permute iterate is reached, after which return nothing for an error code

#three permutation modes: permute (iterative random changes to mix matrix and sources until model lh>contour or iterate limit reached)
#						-(iterates, moves, move_sizes, PWM_shift_range (a Distribution)) for permute params
#						init (iteratively reinitialize sources from priors)
#						-(iterates) for init params
#						merge (iteratively copy a source + mix matrix row from another model in the ensemble until lh>contour or iterate						limit reached)
#						-(iterates) for merpge params
function run_permutation_routine(e::Bayes_IPM_ensemble, param_set::Vector{Tuple{String,Any}}, models_to_permute::Int64, contour::Float64; reset=true)
	for i = 1:models_to_permute
		m_record = rand(e.models)
		m = deserialize(m_record.path)
		original = deepcopy(m)
		for (mode, params) in param_set
			if mode == "permute"
				reset && (m = deepcopy(original))
				permute_model!(m, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
			elseif mode == "merge"
				reset && (m = deepcopy(original))
				merge_model!(e.models, m, contour, e.obs_array,  e.obs_lengths, e.bg_scores, params...)
			elseif mode == "init"
				reset && (m = deepcopy(original))
				reinit_sources!(m, contour,  e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, e.mix_prior, params...)
			else
				@error "Malformed permute mode code! Current supported: \"permute\", \"merge\", \"init\""
			end
			m.log_likelihood > contour && return m
		end
	end
	return nothing
end

function worker_permute(e::Bayes_IPM_ensemble, librarian::Int64, job_chan::RemoteChannel, models_chan::RemoteChannel, param_set::Vector{Tuple{String,Any}}, permute_limit::Int64; reset=true)
	persist=true
	id=myid()
	while persist
		wait(job_chan)
		models = fetch(job_chan)
		models === nothing && (persist=false) && break
		contour, ll_idx = findmin([model.log_Li for model in models])
		deleteat!(models, ll_idx)

		for i=1:permute_limit
			m_record = rand(models)
			job_model = remotecall_fetch(deserialize,librarian,m_record.path)
			original = deepcopy(job_model)

			for (mode, params) in param_set
				if mode == "permute"
					reset && (job_model = deepcopy(original))
					permute_model!(job_model, contour, e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, params...)
				elseif mode == "merge"
					reset && (job_model = deepcopy(original))
					merge_model!(librarian, models, job_model, contour, e.obs_array,  e.obs_lengths, e.bg_scores, params...)
				elseif mode == "init"
					reset && (job_model = deepcopy(original))
					reinit_sources!(job_model, contour,  e.obs_array, e.obs_lengths, e.bg_scores, e.source_priors, e.mix_prior, params...)
				else
					@error "Malformed permute mode code! Current supported: \"permute\", \"merge\", \"init\""
				end
				job_model.log_likelihood > contour && (put!(models_chan, (job_model,id)); break; break)
				wait(job_chan)
				fetch(job_chan)!=models && (break; break) #if the ensemble has changed during the search, update it
			end
		i==permute_limit && (put!(models_chan,nothing);persist=false)#worker to put nothing on channel if it fails to find a model more likely than contour
		end
	end
end
