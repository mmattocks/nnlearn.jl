mutable struct Bayes_IPM_ensemble
	path::String #ensemble models and popped-out posterior samples serialised here
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
Bayes_IPM_ensemble(path::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::Matrix{Float64}, obs::Array{Int64}, source_length_limits; posterior_switch::Bool=true) =
Bayes_IPM_ensemble(
	path,
	assemble_IPMs(path, no_models, source_priors, mix_prior, bg_scores, obs, source_length_limits),
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

Bayes_IPM_ensemble(worker_pool::Vector{Int64}, path::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::Matrix{Float64}, obs::Array{Int64}, source_length_limits; posterior_switch::Bool=true) =
Bayes_IPM_ensemble(
	path,
	distributed_IPM_assembly(worker_pool, path, no_models, source_priors, mix_prior, bg_scores, obs, source_length_limits),
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

function assemble_IPMs(path::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::AbstractArray{Float64}, obs::AbstractArray{Int64}, source_length_limits::UnitRange{Int64})
	ensemble_records = Vector{Model_Record}()
	!isdir(path) && mkpath(path)

	@assert size(obs)[2]==size(bg_scores)[2]

	@showprogress 1 "Assembling IPM ensemble..." for model_no in 1:no_models
		model_path = string(path,'/',model_no)
		if !isfile(model_path)
			model = ICA_PWM_model(string(model_no), source_priors, mix_prior, bg_scores, obs, source_length_limits)
			serialize(model_path, model) #save the model to the ensemble directory
			push!(ensemble_records, Model_Record(model_path,model.log_Li))
		else #interrupted assembly pick up from where we left off
			model = deserialize(model_path)
			push!(ensemble_records, Model_Record(model_path,model.log_Li))
		end
	end

	return ensemble_records
end

function distributed_IPM_assembly(worker_pool::Vector{Int64}, path::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,Float64}, bg_scores::AbstractArray{Float64}, obs::AbstractArray{Int64}, source_length_limits::UnitRange{Int64})
	ensemble_records = Vector{Model_Record}()
	!isdir(path) && mkpath(path)

	@assert size(obs)[2]==size(bg_scores)[2]

    model_chan= RemoteChannel(()->Channel{ICA_PWM_model}(length(worker_pool)))
    job_chan = RemoteChannel(()->Channel{Union{Tuple,Nothing}}(1))
	put!(job_chan,(source_priors, mix_prior, bg_scores, obs, source_length_limits))
	
    for worker in worker_pool
        remote_do(worker_assemble, worker, job_chan, model_chan)
	end
	
	assembly_progress=Progress(no_models, desc="Assembling IPM ensemble...")

	model_counter=check_assembly!(ensemble_records, path, no_models, assembly_progress)

	while model_counter <= no_models
		wait(model_chan)
		candidate=take!(model_chan)
		model = ICA_PWM_model(string(model_counter), candidate.sources, candidate.informed_sources, candidate.source_length_limits, candidate.mix_matrix, candidate.log_Li, candidate.flags)
		model_path=string(path,'/',model_counter)
		serialize(model_path,model)
		push!(ensemble_records, Model_Record(model_path,model.log_Li))
		model_counter+=1
		next!(assembly_progress)
	end

	take!(job_chan),put!(job_chan,nothing)

	return ensemble_records
end
				function check_assembly!(ensemble_records::Vector{Model_Record}, path::String, no_models::Int64, assembly_progress::Progress)
					counter=1
					while counter <= no_models
						model_path=string(path,'/',counter)
						if isfile(model_path)
							model=deserialize(model_path)
							push!(ensemble_records, Model_Record(model_path,model.log_Li))
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
