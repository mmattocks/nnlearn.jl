#### IMPLEMENTATION OF JEFF SKILLINGS' NESTED SAMPLING ALGORITHM ####
function nested_step!(e::Bayes_IPM_ensemble, param_set, permute_limit::Int64)
    N = length(e.models) #number of sample models/particles on the posterior surface
    i = length(e.log_Li) #iterate number, index for last values
    j = i+1 #index for newly pushed values

    #REMOVE OLD LEAST LIKELY MODEL
    ll_contour, least_likely_idx = findmin([model.log_Li for model in e.models])

    Li_model = e.models[least_likely_idx]
    deleteat!(e.models, least_likely_idx)

    e.sample_posterior ? push!(e.retained_posterior_samples, Li_model) : rm(Li_model.path) #if sampling posterior, push the model record to the ensemble's posterior samples vector, otherwise delete the serialised model pointed to by the model record

    #SELECT NEW MODEL, SAVE TO ENSEMBLE DIRECTORY, CREATE RECORD AND PUSH TO ENSEMBLE
    model_selected=false
    while !model_selected
        candidate=run_permutation_routine(e, param_set, permute_limit, ll_contour)
        if !(candidate===nothing)
            model_selected=true
            new_model_record = Model_Record(string(e.ensemble_directory,'/',candidate.name), candidate.log_likelihood);
            push!(e.models, new_model_record);
            serialize(new_model_record.path, candidate)
            e.model_counter +=1
        else
            return 1
        end
    end
         
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

function nested_step!(e::Bayes_IPM_ensemble, model_chan, param_set, permute_limit::Int64,  librarians::Vector{Int64}, worker_pool::Vector{Int64})
    N = length(e.models) #number of sample models/particles on the posterior surface
    i = length(e.log_Li) #iterate number, index for last values
    j = i+1 #index for newly pushed values

    #REMOVE OLD LEAST LIKELY MODEL
    ll_contour, least_likely_idx = findmin([model.log_Li for model in e.models])

    Li_model = e.models[least_likely_idx]
    deleteat!(e.models, least_likely_idx)

    e.sample_posterior ? push!(e.retained_posterior_samples, Li_model) : rm(Li_model.path) #if sampling posterior, push the model record to the ensemble's posterior samples vector, otherwise delete the serialised model pointed to by the model record

    #SELECT NEW MODEL, SAVE TO ENSEMBLE DIRECTORY, CREATE RECORD AND PUSH TO ENSEMBLE
    model_selected=false; worker_persistence=trues(length(worker_pool))
    while !model_selected
        for (x,worker) in enumerate(worker_pool)
            librarian = librarians[Int(ceil(x/length(librarians)))]
            job_model=deserialize((rand(e.models).path))
            remote_do(worker_permute, worker, librarian, e, i, job_model, ll_contour, model_chan, param_set, permute_limit)
        end
        wait(model_chan)
        candidate, iterate = take!(model_chan)

        if !(candidate===nothing)
            if candidate.log_likelihood > ll_contour
                model_selected=true
                new_model_record = Model_Record(string(e.ensemble_directory,'/',candidate.name), candidate.log_likelihood);
                push!(e.models, new_model_record);
                serialize(new_model_record.path, candidate)
                e.model_counter +=1
            end
        else
            @warn "Worker failed to find new model in contour $ll_contour in iterate $i prior to reaching convergence criterion";
            iterate=i && (worker_persistence[findfirst(worker_persistence)]=false)
            !any(worker_persistence) && return 1
        end
    end
         
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

function ns_converge!(e::Bayes_IPM_ensemble, param_set, permute_limit::Int64, evidence_fraction::Float64=.001, verbose::Bool=false)
    N = length(e.models)
    log_frac=log(evidence_fraction)
    
    iterate = length(e.log_Li) #get the iterate from the enemble 
    meter = ProgressNS(e.naive_lh, typemax(Float64), "Nested Sampling: ", 0, iterate)

    while CLHMM.lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= CLHMM.lps(log_frac,e.log_Zi[end])
        iterate = length(e.log_Li) #get the iterate from the enemble 
        warn = nested_step!(e,  param_set, permute_limit) #step the ensemble
        warn == 1 && #"1" passed for warn code means no workers persist; all have hit the permute limit
                (@error "All workers failed to find new models, aborting at current iterate."; return e) #if there is a warning, iust return the ensemble and print info

        iterate += 1

        update!(meter, e.log_Li[end], findmax([model.log_Li for model in e.models])[1], CLHMM.lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]), log(evidence_fraction * exp(e.log_Zi[end])), e.Hi[end], e.log_Zi[end])

        verbose && @info "Iterate: $iterate, contour: $(e.log_Li[end]), max ensemble likelihood: $(findmax([model.log_Li for model in e.models])[1]), convergence val: $(CLHMM.lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end])), covergence criterion: $(log(evidence_fraction * exp(e.log_Zi[end]))), log_Xi:$(e.log_Xi[end]), log_wt:$(e.log_wi[end]), log_liwi:$(e.log_Liwi[end]), log_Z:$(e.log_Zi[end]), H:$(e.Hi[end])"
    end

    final_logZ = logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(1/length(e.models))

    @info "Job done, sampled to convergence. Final logZ $final_logZ"

    return final_logZ
end

    
function ns_converge!(e::Bayes_IPM_ensemble, param_set, permute_limit::Int64, librarians::Vector{Int64}, worker_pool::Vector{Int64}, evidence_fraction::Float64=.001, verbose::Bool=false)
    N = length(e.models)
    log_frac=log(evidence_fraction)
    model_chan= RemoteChannel(()->Channel{Tuple{ICA_PWM_model,Int64}}(length(worker_pool))) #channel to take EM iterates off of
    
    meter = ProgressNS(e.naive_lh, typemax(Float64))

    while CLHMM.lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= CLHMM.lps(log_frac,e.log_Zi[end])
        iterate = length(e.log_Li) #get the iterate from the enemble 
        warn = nested_step!(e, model_chan, param_set, permute_limit, librarians, worker_pool) #step the ensemble
        warn == 1 && #"1" passed for warn code means no workers persist; all have hit the permute limit
                (@error "All workers failed to find new models, aborting at current iterate."; return e) #if there is a warning, iust return the ensemble and print info

        iterate += 1

        update!(meter, e.log_Li[end], findmax([model.log_Li for model in e.models])[1], CLHMM.lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]), log(evidence_fraction * exp(e.log_Zi[end])), e.Hi[end], e.log_Zi[end])

        verbose && @info "Iterate: $iterate, contour: $(e.log_Li[end]), max ensemble likelihood: $(findmax([model.log_Li for model in e.models])[1]), convergence val: $(CLHMM.lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end])), covergence criterion: $(log(evidence_fraction * exp(e.log_Zi[end]))), log_Xi:$(e.log_Xi[end]), log_wt:$(e.log_wi[end]), log_liwi:$(e.log_Liwi[end]), log_Z:$(e.log_Zi[end]), H:$(e.Hi[end])"
    end

    final_logZ = logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(1/length(e.models))

    @info "Job done, sampled to convergence. Final logZ $final_logZ"

    return final_logZ
end