#### IMPLEMENTATION OF JEFF SKILLINGS' NESTED SAMPLING ALGORITHM ####
function nested_step!(e::Bayes_IPM_ensemble, param_set, permute_limit::Int64)
    N = length(e.models) #number of sample models/particles on the posterior surface
    i = length(e.log_Li) #iterate number, index for last values
    j = i+1 #index for newly pushed values

    ll_contour, least_likely_idx = findmin([model.log_Li for model in e.models])

    #REMOVE OLD LEAST LIKELY MODEL
    Li_model = e.models[least_likely_idx]
    deleteat!(e.models, least_likely_idx)

    e.sample_posterior ? push!(e.retained_posterior_samples, Li_model) : rm(Li_model.path) #if sampling posterior, push the model record to the ensemble's posterior samples vector, otherwise delete the serialised model pointed to by the model record

    #SELECT NEW MODEL, SAVE TO ENSEMBLE DIRECTORY, CREATE RECORD AND PUSH TO ENSEMBLE
    model_selected=false; step_report=0
    while !model_selected
        candidate,step_report=run_permutation_routine(e, param_set..., permute_limit, ll_contour)
        if !(candidate===nothing)
            model_selected=true
            new_model_record = Model_Record(string(e.path,'/',e.model_counter), candidate.log_Li);
            push!(e.models, new_model_record);
            final_model=ICA_PWM_model(string(e.model_counter), candidate.sources, candidate.informed_sources, candidate.source_length_limits, candidate.mix_matrix, candidate.log_Li, candidate.flags)
            serialize(new_model_record.path, final_model)
            e.model_counter +=1
        else
            push!(e.models, Li_model)
            return 1, nothing
        end
    end
         
    #UPDATE ENSEMBLE QUANTITIES   
    push!(e.log_Li, minimum([model.log_Li for model in e.models])) #log likelihood of the least likely model - the current ensemble ll contour at Xi
    push!(e.log_Xi, -i/N) #log Xi - crude estimate of the iterate's enclosed prior mass
    push!(e.log_wi, log(exp(e.log_Xi[i]) - exp(e.log_Xi[j]))) #log width of prior mass spanned by the last step
    push!(e.log_Liwi, lps(e.log_Li[j],e.log_wi[j])) #log likelihood + log width = increment of evidence spanned by iterate
    push!(e.log_Zi, logaddexp(e.log_Zi[i],e.log_Liwi[j]))    #log evidence
    #information- dimensionless quantity
    push!(e.Hi, lps(
            (exp(lps(e.log_Liwi[j],-e.log_Zi[j])) * e.log_Li[j]), #term1
            (exp(lps(e.log_Zi[i],-e.log_Zi[j])) * lps(e.Hi[i],e.log_Zi[i])), #term2
            -e.log_Zi[j])) #term3

    return 0, step_report
end

function nested_step!(e::Bayes_IPM_ensemble, model_chan::RemoteChannel, worker_persistence::BitVector)
    N = length(e.models) #number of sample models/particles on the posterior surface
    i = length(e.log_Li) #iterate number, index for last values
    j = i+1 #index for newly pushed values

    #REMOVE OLD LEAST LIKELY MODEL
    ll_contour, least_likely_idx = findmin([model.log_Li for model in e.models])

    Li_model = e.models[least_likely_idx]
    deleteat!(e.models, least_likely_idx)

    e.sample_posterior ? push!(e.retained_posterior_samples, Li_model) : rm(Li_model.path) #if sampling posterior, push the model record to the ensemble's posterior samples vector, otherwise delete the serialised model pointed to by the model record

    #SELECT NEW MODEL, SAVE TO ENSEMBLE DIRECTORY, CREATE RECORD AND PUSH TO ENSEMBLE
    model_selected=false;wk=0;step_report=0
    while !model_selected
        wait(model_chan)
        model_tuple = take!(model_chan)
        if !(model_tuple===nothing)
            candidate,wk,step_report=model_tuple
            if candidate.log_Li > ll_contour
                model_selected=true
                new_model_record = Model_Record(string(e.path,'/',e.model_counter), candidate.log_Li);
                push!(e.models, new_model_record);
                final_model=ICA_PWM_model(string(e.model_counter), candidate.sources, candidate.informed_sources, candidate.source_length_limits, candidate.mix_matrix, candidate.log_Li)
                serialize(new_model_record.path, final_model)
                e.model_counter +=1
            end
        else
            @warn "Worker failed to find new model in contour $ll_contour in iterate $i prior to reaching convergence criterion";
            worker_persistence[findfirst(worker_persistence)]=false
            !any(worker_persistence) && ((push!(e.models, Li_model)); return (1,0,0))
        end
    end
    
    #UPDATE ENSEMBLE QUANTITIES   
    push!(e.log_Li, minimum([model.log_Li for model in e.models])) #log likelihood of the least likely model - the current ensemble ll contour at Xi
    push!(e.log_Xi, -i/N) #log Xi - crude estimate of the iterate's enclosed prior mass
    push!(e.log_wi, log(exp(e.log_Xi[i]) - exp(e.log_Xi[j]))) #log width of prior mass spanned by the last step
    push!(e.log_Liwi, lps(e.log_Li[j],e.log_wi[j])) #log likelihood + log width = increment of evidence spanned by iterate
    push!(e.log_Zi, logaddexp(e.log_Zi[i],e.log_Liwi[j]))    #log evidence

    #information- dimensionless quantity
    push!(e.Hi, lps(
            (exp(lps(e.log_Liwi[j],-e.log_Zi[j])) * e.log_Li[j]), #term1
            (exp(lps(e.log_Zi[i],-e.log_Zi[j])) * lps(e.Hi[i],e.log_Zi[i])), #term2
            -e.log_Zi[j])) #term3

    return 0, wk, step_report
end

function ns_converge!(e::Bayes_IPM_ensemble, param_set, permute_limit::Int64, evidence_fraction::Float64=.001; model_display=1, backup::Tuple{Bool,Int64}=(false,0), verbose::Bool=false)
    N = length(e.models)
    log_frac=log(evidence_fraction)
    
    iterate = length(e.log_Li) #get the iterate from the enemble 
    meter = ProgressNS(e.naive_lh, typemax(Float64), [1], "Nested Sampling::", 0, iterate, no_displayed_srcs=model_display)

    while lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= lps(log_frac,e.log_Zi[end])
        iterate = length(e.log_Li) #get the iterate from the enemble 
        warn,step_report = nested_step!(e, param_set, permute_limit) #step the ensemble
        warn == 1 && #"1" passed for warn code means no workers persist; all have hit the permute limit
                (@error "All workers failed to find new models, aborting at current iterate."; return e) #if there is a warning, iust return the ensemble and print info
        iterate += 1

        backup[1] && iterate%backup[2] == 0 && serialize(string(e.path,'/',"ens"), e) #every backup interval, serialise the ensemble

        e_update_progress(e,meter,log_frac,1,step_report,model_display)        
    end

    final_logZ = logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(1/length(e.models))

    @info "Job done, sampled to convergence. Final logZ $final_logZ"

    return final_logZ
end

    
function ns_converge!(e::Bayes_IPM_ensemble, param_sets, permute_limit::Int64, librarians::Vector{Int64}, worker_pool::Vector{Int64}, evidence_fraction::Float64=.001; model_display::Int64=1, backup::Tuple{Bool,Int64}=(false,0), verbose::Bool=false)
    N = length(e.models)
    log_frac=log(evidence_fraction)

    model_chan= RemoteChannel(()->Channel{Union{Tuple{ICA_PWM_model,Int64, Tuple},Nothing}}(length(worker_pool))) #channel to take EM iterates off of
    job_chan = RemoteChannel(()->Channel{Union{Vector{Model_Record},Nothing}}(1))
    put!(job_chan,e.models)

    worker_persistence=trues(length(worker_pool))

    @assert length(param_sets)==length(worker_pool) "Each worker must have a parameter set!"

    for (x,worker) in enumerate(worker_pool)
        librarian = librarians[Int(ceil(x*(length(librarians)/length(worker_pool))))]
        remote_do(worker_permute, worker, e, librarian, job_chan, model_chan, param_sets[x]..., permute_limit)
    end

    iterate = length(e.log_Li) #get the iterate from the ensemble 
    meter = ProgressNS(e.naive_lh, typemax(Float64), worker_pool, "Nested Sampling::", 0, iterate, no_displayed_srcs=model_display)

    while lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= lps(log_frac,e.log_Zi[end])
        iterate = length(e.log_Li) #get the iterate from the ensemble 
        warn, wk, step_report = nested_step!(e, model_chan, worker_persistence) #step the ensemble
        warn == 1 && #"1" passed for warn code means no workers persist; all have hit the permute limit
                (@error "All workers failed to find new models, aborting at current iterate."; return e) #if there is a warning, iust return the ensemble and print info
        iterate += 1

        take!(job_chan); put!(job_chan,e.models)

        backup[1] && iterate%backup[2] == 0 && serialize(string(e.path,'/',"ens"), e) #every backup interval, serialise the ensemble

        e_update_progress(e,meter,log_frac,wk,step_report,model_display)        
    end

    take!(job_chan); put!(job_chan, nothing)

    final_logZ = logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(1/length(e.models))

    @info "Job done, sampled to convergence. Final logZ $final_logZ"

    return final_logZ
end

                function e_update_progress(e,meter,log_frac,wk,step_report,model_display)
                    Li_vec=[model.log_Li for model in e.models]
                    if model_display>0
                        max_Li,max_idx=findmax(Li_vec)
                        max_model=deserialize(e.models[max_idx].path)
                        update!(meter, e.log_Li[end], max_Li, lps(max_Li,  e.log_Xi[end]), lps(log_frac,e.log_Zi[end]), e.Hi[end], Li_vec, wk, step_report...,sources=max_model.sources,bitmatrix=max_model.mix_matrix)
                    else
                        update!(meter, e.log_Li[end], max_Li, lps(max_Li,  e.log_Xi[end]), lps(log_frac,e.log_Zi[end]), e.Hi[end], Li_vec, wk, step_report...)
                    end
                end