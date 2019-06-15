__precompile__()

"""
Functions to learn a metamotif model of nucleosome positions by nested sampling
"""

module nnlearn
    using BGHMM, BioSequences, DataFrames, MS_HMMBase, ProgressMeter, SimpleSparseArrays
    import Distributed: RemoteChannel
    import Distributions: Dirichlet, Categorical
    import StatsFuns: logsumexp
    import Random: rand

    function make_padded_df(position_fasta::String, gff3_path::String, genome_path::String, genome_index_path::String, pad::Int64)
        position_reader = BioSequences.FASTA.Reader(open((position_fasta),"r"))
        genome_reader = open(BioSequences.FASTA.Reader, genome_path, index=genome_index_path)
        scaffold_df = BGHMM.build_scaffold_df(gff3_path)
        position_df = DataFrame(SeqID = String[], Start=Int64[], End=Int64[], PadSeq = DNASequence[], PadStart=Int64[], SeqOffset=Int64[])
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

                push!(position_df, [scaffold, pos_start, pos_end, padded_seq, pad_start, seq_offset])
            end
        end

        close(position_reader)
        close(genome_reader)
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
