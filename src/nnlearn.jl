__precompile__()

"""
Functions to learn a metamotif model of nucleosome positions by nested sampling
"""

module nnlearn
    using BGHMM, BioSequences, DataFrames, Distributed, Distributions, MS_HMMBase, ProgressMeter, Serialization, SimpleSparseArrays
    import StatsFuns: logaddexp, logsumexp #both are needed as logsumexp for two terms is deprecated
    import Random: rand

    function make_padded_df(position_fasta::String, gff3_path::String, genome_path::String, genome_index_path::String, pad::Int64)
        position_reader = BioSequences.FASTA.Reader(open((position_fasta),"r"))
        genome_reader = open(BioSequences.FASTA.Reader, genome_path, index=genome_index_path)
        scaffold_df = BGHMM.build_scaffold_df(gff3_path)
        position_df = DataFrame(SeqID = String[], Start=Int64[], End=Int64[], PadSeq = DNASequence[], PadStart=Int64[], RelStart=Int64[], SeqOffset=Int64[])
        scaffold_seq_dict = BGHMM.build_scaffold_seq_dict(genome_path, genome_index_path)

        for entry in position_reader
            scaffold = BioSequences.FASTA.identifier(entry)

            if scaffold != "MT"
                desc_array = split(BioSequences.FASTA.description(entry))
                pos_start = parse(Int64, desc_array[2])
                pos_end = parse(Int64, desc_array[4])
                scaffold_end = scaffold_df.End[findfirst(isequal(scaffold), scaffold_df.SeqID)]

                pad_start=max(1,pos_start-pad)
                pad_length= pos_start - pad_start
                seq_offset = pad - pad_length
                padded_seq = BGHMM.fetch_sequence(scaffold, scaffold_seq_dict, pad_start, pos_end, '+')

                if !hasambiguity(padded_seq)
                    push!(position_df, [scaffold, pos_start, pos_end, padded_seq, pad_start, pad_length, seq_offset])
                end
            end
        end

        close(position_reader)
        close(genome_reader)
        return position_df
    end

    function observation_setup(position_df::DataFrame, pad::Int64)
        offsets = position_df.SeqOffset
        order_seqs = BGHMM.get_order_n_seqs(position_df.PadSeq, 0)
        coded_seqs = BGHMM.code_seqs(order_seqs, offsets)

        return coded_seqs, offsets
    end

    function assemble_source_priors(no_sources::Int64, wm_samples, uninform_length_range::UnitRange{Int64}, unspecified_wm_prior=Dirichlet(ones(4)/4)) #unless otherwise specified, fill non-wm_samples-specified priors with uninformative priors of random lengths in the specified range
        source_priors = Vector{Vector{Dirichlet{Float64}}}()
        for source in 1:no_sources
            if source <= length(wm_samples)
                push!(source_priors, estimate_dirichlet_prior_on_wm(wm_samples[source]))
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

                function estimate_dirichlet_prior_on_wm(wm::Matrix{Float64}; wt::Float64=3.0)
                    prior = Vector{Dirichlet{Float64}}()
                    for position in 1:size(wm)[1]
                        push!(prior, Dirichlet(wm[position,:].*wt))
                    end
                    return prior
                end

    include("bghmm_likelihood.jl")
    include("ICA_PWM_model.jl")
    include("Bayes_IPM_ensemble.jl")
    include("nested_sampler.jl")

end # module
