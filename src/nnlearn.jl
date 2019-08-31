__precompile__()

"""
Functions to learn a metamotif model of nucleosome positions by nested sampling
"""

module nnlearn
    using BenchmarkTools, BGHMM, BioSequences, DataFrames, Distributed, Distributions, MS_HMMBase, ProgressMeter, Serialization
    import StatsFuns: logaddexp, logsumexp #both are needed as logsumexp for two terms is deprecated
    import Random: rand

    mutable struct Model_Record #record struct to associate a log_Li with a saved, calculated model
        path::String
        log_Li::Float64
    end

    function observation_setup(position_df::DataFrame, pad::Int64)
        offsets = position_df.SeqOffset
        order_seqs = BGHMM.get_order_n_seqs(position_df.PadSeq, 0)
        coded_seqs = BGHMM.code_seqs(order_seqs, position_df.SeqOffset)

        return coded_seqs, position_df.SeqOffset
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
                        @assert MS_HMMBase.isprobvec(wm[i,:])
                    end
                    prior = Vector{Dirichlet{Float64}}()
                    for position in 1:size(wm)[1]
                        push!(prior, Dirichlet(wm[position,:].*wt))
                    end
                    return prior
                end

    include("ICA_PWM_model.jl")
    include("Bayes_IPM_ensemble.jl")
    include("nested_sampler.jl")
    include("model_display.jl")
    include("performance.jl")
end # module
