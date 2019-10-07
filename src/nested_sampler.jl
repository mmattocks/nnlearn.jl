#### IMPLEMENTATION OF JEFF SKILLINGS' NESTED SAMPLING ALGORITHM, VERY ROUGHLY FOLLOWING THOMAS DOWNS' NMICA ####
function nested_step!(e::Bayes_IPM_ensemble, perm_params::Vector{Tuple{String,Any}},models_to_permute::Int64)
    N = length(e.models) #number of sample models/particles on the posterior surface
    i = length(e.log_Li) #iterate number, index for last values
    j = i+1 #index for newly pushed values

    #REMOVE OLD LEAST LIKELY MODEL
    ll_contour, least_likely_idx = findmin([model.log_Li for model in e.models])

    Li_model = e.models[least_likely_idx]
    deleteat!(e.models, least_likely_idx)

    e.sample_posterior ? push!(e.retained_posterior_samples, Li_model) : rm(Li_model.path) #if sampling posterior, push the model record to the ensemble's posterior samples vector, otherwise delete the serialised model pointed to by the model record

    #SELECT NEW MODEL, SAVE TO ENSEMBLE DIRECTORY, CREATE RECORD AND PUSH TO ENSEMBLE
    @info "Permuting..."
    new_model = run_permutation_routine(e, perm_params, models_to_permute, ll_contour)
    new_model === nothing ? (@error "Failure to find new model in current likelihood contour $ll_contour, iterate $i, prior to reaching convergence criterion"; return 1) : new_model_record = Model_Record(string(e.ensemble_directory,new_model.name), new_model.log_likelihood)
    push!(e.models, new_model_record)
    serialize(new_model_record.path, new_model)
    e.model_counter +=1

    #UPDATE ENSEMBLE QUANTITIES   
    push!(e.log_Li, minimum([model.log_Li for model in e.models])) #log likelihood of the least likely model - the current ensemble ll contour at Xi
    push!(e.log_Xi, -i/N) #log Xi - crude estimate of the iterate's enclosed prior mass
    push!(e.log_wi, log(exp(e.log_Xi[i]) - exp(e.log_Xi[j]))) #log width of prior mass spanned by the last step
    push!(e.log_Liwi, CLHMM.lps(e.log_Li[j],e.log_wi[j])) #log likelihood + log width = increment of evidence spanned by iterate
    push!(e.log_Zi, logaddexp(e.log_Zi[i],e.log_Liwi[j]))    #log evidence

    #information- dimensionless quantity
    push!(e.Hi, CLHMM.lps(
            (exp(CLHMM.lps(e.log_Liwi[j],-e.log_Zi[j])) * e.log_Li[j]), #term1
            (exp(CLHMM.lps(e.log_Zi[i],-e.log_Zi[j])) * CLHMM.lps(e.Hi[i],e.log_Zi[i])), #term2
            -e.log_Zi[j])) #term3

    return 0
end
    
function nested_sample_posterior_to_convergence!(e::Bayes_IPM_ensemble, permute_params, permute_limit, evidence_fraction::Float64=.001; verbose::Bool=true)
    N = length(e.models)
    #est_std_logZ = sqrt(e.Hi[end]/N)
    #Lmax_model_idx = findmax([model.log_Li[1] for model in e.models])[2]

    #while e.models[Lmax_model_idx].log_likelihood[1] * e.log_Xi[end] >= evidence_fraction * (e.log_Zi[end]-2*est_std_logZ) #latter bracketed term is 2σ lower estimate for Zo
    iterate=0
    #@info "$iterate <-iterate thresh-> $(N*e.Hi[end]*10)"
    while iterate <= N * e.Hi[end] * 10 
        iterate = length(e.log_Li) #get the iterate from the enemble 
        warn = nested_step!(e, permute_params, permute_limit) #step the ensemble
        warn == 1 && #"1" passed for warn code means nested_step failed to find permutation inside llh contour
                return e #if there is a warning, iust return the ensemble
        verbose && @info "Iterate: $iterate, contour: $(e.log_Li[end]), log_Xi:$(e.log_Xi[end]), log_wt:$(e.log_wi[end]) log_liwi:$(e.log_Liwi[end]), log_Z:$(e.log_Zi[end]), H:$(e.Hi[end])"
        iterate += 1
    end

    final_logZ = logsumexp([model.log_Li[1] for model in e.models]) +  e.log_Xi[iterate] - log(1/length(e.models))

    @info "Job done, sampled to convergence. Final logZ $final_logZ"

    return final_logZ
end