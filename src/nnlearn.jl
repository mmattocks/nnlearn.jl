__precompile__()

"""
Functions to learn a metamotif model of nucleosome positions by nested sampling
"""

module nnlearn
    using BenchmarkTools, BioSequences, DataFrames, Distributed, Distributions, ProgressMeter, Serialization, UnicodePlots, JLD2
    import ProgressMeter: AbstractProgress
    import Printf: @sprintf
    import StatsFuns: logaddexp, logsumexp #both are needed as logsumexp for two terms is deprecated
    import Random: rand, seed!, shuffle!
    import Distances: euclidean

    mutable struct Model_Record #record struct to associate a log_Li with a saved, calculated model
        path::String
        log_Li::Float64
    end

    function read_fa_wms_tr(path::String)
        wms=Vector{Matrix{Float64}}()
        wm=zeros(1,4)
        f=open(path)
        for line in eachline(f)
            prefix=line[1:2]
            prefix == "01" && (wm=transpose([parse(Float64,i) for i in split(line)[2:end]]))
            prefix != "01" && prefix != "NA" && prefix != "PO" && prefix != "//" && (wm=vcat(wm, transpose([parse(Float64,i) for i in split(line)[2:end]])))
            prefix == "//" && push!(wms, wm)
        end
        return wms
    end

    #wm_samples are in decimal probability space, not log space
    function assemble_source_priors(no_sources::Int64, wm_samples::Vector{Matrix{Float64}}, prior_wt::Float64, uninform_length_range::UnitRange{Int64}, unspecified_wm_prior=Dirichlet(ones(4)/4)) #estimate a dirichlet prior on wm_samples inputs; unless otherwise specified, fill non-wm_samples-specified priors with uninformative priors of random lengths in the specified range
        source_priors = Vector{Vector{Dirichlet{Float64}}}()
        for source in 1:no_sources
            if source <= length(wm_samples)
                push!(source_priors, estimate_dirichlet_prior_on_wm(wm_samples[source], prior_wt))
            else
                uninformative_prior = Vector{Dirichlet{Float64}}()
                for pos in 1:rand(uninform_length_range)
                    push!(uninformative_prior, unspecified_wm_prior)
                end
                push!(source_priors, uninformative_prior)
            end
        end
        return source_priors
    end

                function estimate_dirichlet_prior_on_wm(wm::Matrix{Float64}, wt::Float64)
                    for i in 1:size(wm)[1]
                        !(isprobvec(wm[i,:])) && throw(DomainError("Bad weight vec supplied to estimate_dirichlet_prior_on_wm! $(wm[i,:])"))
                    end
                    prior = Vector{Dirichlet{Float64}}()
                    for position in 1:size(wm)[1]
                        normvec=wm[position,:]
                        zero_idxs=findall(isequal(0.),wm[position,:])
                        normvec[zero_idxs].+=10^-99
                        push!(prior, Dirichlet(normvec.*wt))
                    end
                    return prior
                end

    function cluster_mix_prior!(df::DataFrame, wms::Vector{Matrix{Float64}})
        mix=falses(size(df,1),length(wms))
        for (o, row) in enumerate(eachrow(df))
            row.cluster != 0 && (mix[o,row.cluster]=true)
        end
        
        represented_sources=unique(df.cluster)
        wms=wms[represented_sources]
        return mix[:,represented_sources]
    end
    
    #subfuncs to handle sums of log probabilities that may include -Inf (ie p=0), returning -Inf in this case rather than NaNs
    function lps(adjuvants::AbstractArray)
        prob = sum(adjuvants) ; isnan(prob) ? - Inf : prob
    end
    
    function lps(base, adjuvants...)::Float64
        prob = base+sum(adjuvants) ; isnan(prob) ? -Inf : prob
    end

    include("ICA_PWM_model.jl")
    include("Bayes_IPM_ensemble.jl")
    include("nested_sampler.jl")
    include("model_display.jl")
    include("ns_progressmeter.jl")
    include("ensemble_utils.jl")
    include("model_permutation.jl")
    #include("performance.jl")
end # module
