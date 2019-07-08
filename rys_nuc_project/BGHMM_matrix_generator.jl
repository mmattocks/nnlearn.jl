#JOB FILEPATHS
Sys.islinux() ? danio_gff_path = "/media/main/Bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3" : danio_gff_path = "F:\\PhD\\seq\\GRCz11\\Danio_rerio.GRCz11.94.gff3"
Sys.islinux() ? selected_hmm_output = "/media/main/Bench/PhD/NGS_binaries/BGHMM/selected_BGHMMs" : selected_hmm_output = "F:\\PhD\\NGS_binaries\\BGHMM\\selected_BGHMMs"
Sys.islinux() ? sib_seq_fasta = "/media/main/Bench/PhD/git/nnlearn/rys_nuc_project/sib_nuc_position_sequences.fa" : sib_seq_fasta = "F:\\PhD\\git\\nnlearn\\rys_nuc_project\\sib_nuc_position_sequences.fa"

Sys.islinux() ? danio_genome_path = "/media/main/Bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna" : danio_genome_path = "F:\\PhD\\seq\\GRCz11\\GCA_000002035.4_GRCz11_genomic.fna"
Sys.islinux() ? danio_gen_index_path = "/media/main/Bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna.fai" : danio_gen_index_path = "F:\\PhD\\seq\\GRCz11\\GCA_000002035.4_GRCz11_genomic.fna.fai"

Sys.islinux() ? position_df_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_positions" : position_df_binary = "F:\\PhD\\NGS_binaries\\nnlearn\\BGHMM_sib_positions"
Sys.islinux() ? matrix_output = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix" : matrix_output = "F:\\PhD\\NGS_binaries\\nnlearn\\BGHMM_sib_matrix"


@info "Loading master libraries..."
using Distributed, Serialization, ProgressMeter, Distributions

#JOB CONSTANTS
const BGHMM_lhs= RemoteChannel(()->Channel{Tuple}(Inf)) #channel to take partitioned BGHMM subsequence likelihoods from
const position_size = 141

# #DISTRIBUTED CLUSTER CONSTANTS
# const remote_machine = "10.0.0.12"
const no_local_processes = 10
const no_remote_processes = 0
# #SETUP DISTRIBUTED MATRIX LH CALCULATORS
@info "Spawning workers..."
addprocs(no_local_processes, topology=:master_worker)
# #addprocs([(remote_machine,no_remote_processes)], tunnel=true, topology=:master_worker)
pool_size = no_remote_processes + no_local_processes
worker_pool = [i for i in 2:pool_size+1]

@info "Loading worker libraries everywhere..."
@everywhere using BGHMM, BioSequences, nnlearn, Revise, MS_HMMBase

@info "Constructing position dataframe from file at $sib_seq_fasta..."
sib_position_df = nnlearn.make_padded_df(sib_seq_fasta, danio_gff_path, danio_genome_path, danio_gen_index_path, position_size-1)

@info "Masking positions by genome partition and strand, then formatting observations..."
BGHMM.add_partition_masks!(sib_position_df, danio_gff_path)

@info "Setting up for BGHMM likelihood calculations..."
BGHMM_dict = deserialize(selected_hmm_output)
BGHMM_likelihood_queue, jobcount, lh_matrix_size = nnlearn.queue_BGHMM_likelihood_calcs(sib_position_df, BGHMM_dict)

@info "Distributing BGHMM likelihood jobs to workers..."
for worker in worker_pool
    remote_do(nnlearn.process_BGHMM_likelihood_queue!, worker, BGHMM_likelihood_queue, BGHMM_lhs, BGHMM_dict)
end

BGHMM_lh_matrix = zeros(lh_matrix_size) #T, Strand, O
@showprogress 1 "Overall batch progress:" 1 for job in jobcount:-1:1
    @debug "$job lh jobs remaining"
    wait(BGHMM_lhs)
    jobid, frag_lhs = take!(BGHMM_lhs)
    (offset, frag_start, o, partition, strand) = jobid
    if strand == -1
        frag_lhs = reverse(frag_lhs)
    end #positive, unstranded frags  are inserted as-is
    BGHMM_lh_matrix[offset+frag_start:offset+frag_start+length(frag_lhs)-1,o] = frag_lhs
end

@info "Serializing matrix to $matrix_output and position dataframe to $position_df_binary..."
serialize(position_df_binary, sib_position_df)
serialize(matrix_output, BGHMM_lh_matrix)
@info "Job done."