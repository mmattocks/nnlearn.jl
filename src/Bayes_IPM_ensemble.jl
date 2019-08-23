mutable struct Bayes_IPM_ensemble
	ensemble_directory::String #ensemble models and popped-out posterior samples serialised here
	models::Vector{Model_Record} #ensemble keeps paths to serialised models and their likelihood tuples rather than keeping the models in memory

	log_Li::Vector{Float64} #likelihood of lowest-ranked model at iterate i
	log_Xi::Vector{Float64} #amt of prior mass included in ensemble countour at Li
	log_wi::Vector{Float64} #width+ of prior mass covered in this source_stop
	log_Liwi::Vector{Float64} #evidentiary weight of the iterate
	log_Zi::Vector{Float64} #ensemble evidence
	Hi::Vector{Float64} #ensemble information

	obs_array::Matrix{Int64} #observations
	position_size::Int64 #nucleosome positions called in danpos are all the same length
	offsets::Vector{Int64} #first base of obs sequence relative to the first base of the overall observation table (for seqs at 5' scaffold boundaries with truncated 5' pads)

	source_priors::Vector{Vector{Dirichlet{Float64}}} #source pwm priors
	mix_prior::Float64 #prior on %age of observations that any given source contributes to

	bg_scores::Matrix{Float64} #precalculated background HMM scores

	sample_posterior::Bool
	retained_posterior_samples::Vector{Model_Record} #list of posterior sample filenames

	model_counter::Int64
end

####Bayes_IPM_ensemble FUNCTIONS
Bayes_IPM_ensemble(ensemble_directory::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::Matrix{Float64}, obs::Array{Int64}, position_size::Int64, rel_starts::Vector{Int64}, source_length_ranges; posterior_switch::Bool=true) =
Bayes_IPM_ensemble(
	ensemble_directory,
	assemble_IPMs(ensemble_directory, no_models, source_priors, mix_prior, bg_scores, obs, position_size, rel_starts, source_length_ranges),
	[-Inf], #L0 = 0
	[0], #ie exp(0) = all of the prior is covered
	[-Inf], #w0 = 0
	[-Inf], #Liwi0 = 0
	[-1e300], #Z0 = 0
	[0], #H0 = 0,
	obs,
	position_size,
	rel_starts,
	source_priors,
	mix_prior,
	bg_scores, #precalculated background score
	posterior_switch,
	Vector{String}(),
	no_models+1)

function assemble_IPMs(ensemble_directory::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, obs::AbstractArray{Int64}, position_size::Int64, offsets::Vector{Int64}, source_length_ranges::UnitRange{Int64})
	ensemble_records = Vector{Model_Record}()
	@showprogress 1 "Assembling ICA PWM model ensemble..." for model_no in 1:no_models
		model = ICA_PWM_model(string(model_no), source_priors, mix_prior, bg_scores, obs, position_size, offsets, source_length_ranges)
		serialize(string(ensemble_directory,model_no), model) #save the model to the ensemble directory
		push!(ensemble_records, Model_Record(string(ensemble_directory,model_no),model.log_likelihood))
	end

	return ensemble_records
end

#permutation routine function- 
#general logic: receive array of permutation parameters, until a model more likely than the least is found:
#randomly select a model from the ensemble (the least likely having been removed by this point), then sample new models by permuting with each of hte given parameter sets until a model more likely than the current contour is found
#if none is found for the candidate model, move on to another candidate until the models_to_permute iterate is reached, after which return nothing for an error code

#three permutation modes: permute (iterative random changes to mix matrix and sources until model lh>contour or iterate limit reached)
#						-(moves, move_sizes, PWM_shift_range, iterates) for permute params
#						init (iteratively reinitialize sources from priors)
#						-(iterates) for init params
#						merge (iteratively copy a source + mix matrix row from another model in the ensemble until lh>contour or iterate						limit reached)
#						-(iterates) for merge params
function run_permutation_routine(e::Bayes_IPM_ensemble, param_set::Vector{Tuple{Tuple{Int64,Vector{Float64},Distributions.Uniform{Float64}},Int64}}, models_to_permute::Int64, contour::Float64)
	@showprogress 2 "Permuting models, search contour $contour: " for i = 1:models_to_permute
		m_record = rand(e.models)
		m = deserialize(m_record.path)
		for (mode, params) in param_set
			if mode == "permute"
				permute_model!(m, e.model_counter, contour, e.obs_array, e.bg_scores, e.position_size, e.offsets, e.source_priors, params...)
			elseif mode == "merge"
				merge_model!(e.models, m, e.model_counter, contour, e.obs_array, e.bg_scores, e.position_size, e.offsets, params...)
			elseif mode == "init"
				reinit_sources!(m, e.model_counter, contour,  e.obs_array, e.bg_scores, e.position_size, e.offsets, e.source_priors, e.mix_prior, params...)
			else
				@error "Malformed permute mode code! Current supported: \"permute\", \"merge\", \"init\""
			end
			@debug "new model ll: $(m.log_likelihood[1]), contour: $contour"
			m.log_likelihood > contour && return m
		end
	end
	return nothing
end