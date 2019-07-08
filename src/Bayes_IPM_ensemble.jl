mutable struct Model_Record #record struct to associate a log_Li tuple with a saved, calculated model
    path::String
    log_Li::Tuple{Float64, Float64}
end

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

    priors::Vector{Vector{Dirichlet{Float64}}} #source pwm priors

    bg_scores::Matrix{Float64} #precalculated background HMM scores

    sample_posterior::Bool
    retained_posterior_samples::Vector{Model_Record} #list of posterior sample filenames

    model_counter::Int64
end

####Bayes_IPM_ensemble FUNCTIONS
Bayes_IPM_ensemble(ensemble_directory::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::Matrix{Float64}, obs::Array{Int64}, position_size::Int64, rel_starts::Vector{Int64}, source_length_ranges; posterior_switch::Bool=true) =
Bayes_IPM_ensemble(
    ensemble_directory,
    assemble_IPMs(ensemble_directory, no_models, source_priors, mix_prior, view(bg_scores,:,:), view(obs,:,:), position_size, rel_starts, source_length_ranges),
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

#distributed functions
Bayes_IPM_ensemble(jobs_chan::RemoteChannel, results_chan::RemoteChannel, ensemble_directory::String, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::Matrix{Float64}, obs::Array{Int64}, position_size::Int64, rel_starts::Vector{Int64}, source_length_ranges; posterior_switch::Bool=true) =
Bayes_IPM_ensemble(
    ensemble_directory,
    assemble_IPMs(ensemble_directory, jobs_chan, results_chan, no_models, source_priors, mix_prior, view(bg_scores,:,:), view(obs,:,:), position_size, rel_starts, source_length_ranges),
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
   bg_scores, #precalculated background score
   posterior_switch,
   Vector{String}(),
   no_models+1)

function assemble_IPMs(ensemble_directory::String, jobs_chan::RemoteChannel, results_chan::RemoteChannel, no_models::Int64, source_priors::Vector{Vector{Dirichlet{Float64}}}, mix_prior::Float64, bg_scores::AbstractArray{Float64}, obs::AbstractArray{Int64}, position_size::Int64, offsets::Vector{Int64}, source_length_ranges::UnitRange{Int64})
    
    ensemble_records = Vector{Model_Record}()
    @showprogress 1 "Assembling ICA PWM model ensemble..." for model_no in 1:no_models
        model = ICA_PWM_model(jobs_chan, results_chan, string(model_no), source_priors, mix_prior, bg_scores, obs, position_size, offsets, source_length_ranges)
        @info "Saving model..."
        serialize(string(ensemble_directory,model_no), model) #save the model to the ensemble directory
        @info "Saved model!"
        push!(ensemble_records, Model_Record(string(ensemble_directory,model_no),model.log_likelihood))
    end
    take!(jobs_chan) #clear the init job from jobs_chan when we have enough models

    return ensemble_records #file the model records in the ensemble struct
end

function ICA_PWM_worker(jobs_chan::RemoteChannel, results_chan::RemoteChannel)
    while true
        wait(jobs_chan)
        instruction = fetch(jobs_chan)
        if typeof(instruction[1]) <: Bayes_IPM_ensemble
            model = run_permutation_routine(instruction)
            model != nothing ? put!(results_chan, model) : 
                (put!(results_chan, "TIMEOUT"); @error "Failure to find new model in current likelihood contour $(instruction[1].log_Li), iterate $(length(instruction[1].log_Li)), prior to convergence, worker $(myid())")
        elseif instruction[1] == "init"
            put!(results_chan, ICA_PWM_model(instruction[2]...))
        else
            @error "Malformed ICA_PWM_worker job instruction, worker $(myid()), instruction $instruction"
        end
    end
end

#permutation routine function- 
#general logic: receive array of permutation parameters. 
#until a model more likely than the least is found:
#randomly select a model from the ensemble that isn't the least likely one, then sample new models by permuting with each of hte given parameter sets the specified number of times. 
#if none is found for the candidate model, move on to another candidate until the models_to_permute iterate is reached, after which return nothing

#((moves, move_sizes, PWM_shift_range), times_to_apply) for params
function run_permutation_routine(e::Bayes_IPM_ensemble, param_set::Vector{Tuple{Tuple{Int64,Vector{Float64},Distributions.Uniform{Float64}},Int64}}, models_to_permute::Int64, contour::Float64)
    @showprogress 2 "Permuting model $i to find candidate inside contour $contour:" for i = 1:models_to_permute
        m_record = rand(e.models)
        m = deserialize(m_record.path)
        for (params, attempts) in param_set
            attempt = 1
            while attempt <= attempts
                permute_model!(m, e.model_counter, e.obs_array, e.bg_scores, e.position_size, e.offsets, e.priors, params...)
                @debug "new model ll: $(m.log_likelihood[1]), contour: $contour"
                if m.log_likelihood[1] > contour
                    return m
                end
                attempt += 1
            end
        end
    end
    return nothing
end

# function run_permutation_routine(e::Bayes_IPM_ensemble, param_set::Vector{Tuple{Tuple{Int64,Vector{Float64},Distributions.Uniform{Float64}},Int64}}, models_to_permute::Int64, contour::Float64, jobs_chan::RemoteChannel=nothing, results_chan::RemoteChannel=nothing)
#     for i = 1:models_to_permute
#         m_record = rand(e.models)
#         m = deserialize(m_record.path)
#         for (params, attempts) in param_set
#             attempt = 1
#             while attempt <= attempts
#                 permute_model!(jobs_chan, results_chan, m, e.model_counter, e.obs_array, e.bg_scores, e.position_size, e.offsets, e.priors, params...)
#                 @debug "new model ll: $(m.log_likelihood[1]), contour: $contour"
#                 if m.log_likelihood[1] > contour
#                     return m
#                 end
#                 attempt += 1
#             end
#         end
#     end
#     return nothing
# end