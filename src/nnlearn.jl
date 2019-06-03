module nnlearn
    import BGHMM: get_order_n_seqs, code_seqs
    import BioSequences: DNASequence, FASTA.Reader
    import DataFrames: DataFrame
    import Distributions: Categorical
    import HMM: MS_HMMBase


function obs_set_BGHMM_likelihood(BGHMM_dict::Dict{String, HMM}, observations::Matrix{Int64})#numerically code the sequences in trainable format

end

function make_position_df(fasta_reader::BioSequences.FASTA.Reader)
    position_df = DataFrame(Sequence = DNASequence[], Scaffold = String[], Start = Int64[], End = Int64[], Smt = Float64[], Fuzziness = Float64[])
    for entry in fasta_reader
        id = split(entry.identfier)
        sequence = entry.sequence
        push!(position_df, [sequence, id[1], id[3], id[5], id[9], id[11]])
    end
    return position_df
end

function add_partition_mask(position_df::DataFrame, gff3_path::String, perigenic_pad::Int64=500)
    partition_coords_dict = BGHMM.partition_genome_coordinates(gff3_path, perigenic_pad)
    maskcol = Vector{Vector{Int64}}()

    for scaffold_subframe in groupby(position_df, :Sequence)

end



function observation_setup(position_df::DataFrame)
    order_seqs = get_order_n_seqs(position_df.Sequence,0)
    coded_seqs = code_seqs(order_seqs)
    BGHMM_mask = get_partition_mask(position_df)

    return coded_seqs
end




#function to provide a log probability score for a sequence given a weight matrix
function seqScorebyWM(logWM::Matrix, seq::Vector{Int})
    #input WM must be in logarithm form
    score::Float64 = 0.0
    @assert size(logWM)[1] == length(seq)
    for position in 1:length(seq)
        score += logWM[position,seq[position]] #access the weight matrix by the numerical position and the integer value of the coded seq at that position
    end
    return score
end

#sample a weight matrix of a given length from one dirichlet prior applying to each position (uninformative by default)
function sampleWM(length::Int, no_emission_symbols::Int=4, prior::Dirichlet=Dirichlet(ones(no_emission_symbols)/no_emission_symbols))
    wm = zeros(length,no_emission_symbols)
    for position in 1:length
        wm[position,:] = rand(prior)
    end
    return wm
end

end # module
