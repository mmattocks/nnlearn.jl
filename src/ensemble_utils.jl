function reset_ensemble(e::Bayes_IPM_ensemble)
    new_e=deepcopy(e)
    for i in 1:length(e.models)
        if string(i) in [basename(record.path) for record in e.models]
            new_e.models[i]=e.models[findfirst(isequal(string(i)), [basename(record.path) for record in e.models])]
        else
            new_e.models[i]=e.retained_posterior_samples[findfirst(isequal(string(i)), [basename(record.path) for record in e.retained_posterior_samples])]
        end
    end
    return new_e
end