__precompile__()

"""
Functions to learn a metamotif model of nucleosome positions by nested sampling
"""

module nnlearn
    using BGHMM, BioSequences, DataFrames, MS_HMMBase, ProgressMeter
    import Distributed: RemoteChannel
    import Distributions: Dirichlet, Categorical

    function make_position_df(fasta_reader::BioSequences.FASTA.Reader)
        position_df = DataFrame(Sequence = DNASequence[], Scaffold = String[], Start = Int64[], End = Int64[], Smt = Float64[], Fuzziness = Float64[])
        for entry in fasta_reader
            sequence = DNASequence(BioSequences.FASTA.sequence(entry))
            scaffold = BioSequences.FASTA.identifier(entry)
            desc_array = split(BioSequences.FASTA.description(entry))

            push!(position_df, [sequence, scaffold, parse(Int64, desc_array[2]), parse(Int64, desc_array[4]), parse(Float64, desc_array[8]), parse(Float64, desc_array[10])])
        end
        return position_df
    end

    function observation_setup(position_df::DataFrame)
        order_seqs = BGHMM.get_order_n_seqs(position_df.Sequence,0)
        coded_seqs = BGHMM.code_seqs(order_seqs)
        BGHMM_mask = BGHMM.get_mask_matrix(position_df.Mask)

        return coded_seqs, BGHMM_mask
    end

    #sample a weight matrix of a given length from one dirichlet prior applying to each position (uninformative by default)
    function sampleWM(length::Int, no_emission_symbols::Int=4, prior::Dirichlet=Dirichlet(ones(no_emission_symbols)/no_emission_symbols))
        wm = zeros(length,no_emission_symbols)
        for position in 1:length
            wm[position,:] = rand(prior)
        end
        return wm
    end

    include("likelihood_functions.jl")
end # module
