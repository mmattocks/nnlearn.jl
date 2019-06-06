__precompile__()

"""
Functions to learn a metamotif model of nucleosome positions by nested sampling
"""

module nnlearn
    using BGHMM, BioSequences, DataFrames, MS_HMMBase, ProgressMeter
    import Distributed: RemoteChannel
    import Distributions: Dirichlet, Categorical
    import StatsFuns: logsumexp
    import Random: rand

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



    include("bghmm_likelihood.jl")
    include("pwm.jl")
end # module
