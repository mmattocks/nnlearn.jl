#JOB FILEPATHS
Sys.islinux() ? danio_gff_path = "/media/main/Bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3" : danio_gff_path = "F:\\PhD\\seq\\GRCz11\\Danio_rerio.GRCz11.94.gff3"
Sys.islinux() ? selected_hmm_output = "/media/main/Bench/PhD/NGS_binaries/BGHMM/selected_BGHMMs" : selected_hmm_output = "F:\\PhD\\NGS_binaries\\BGHMM\\selected_BGHMMs"
Sys.islinux() ? sib_seq_fasta = "/media/main/Bench/PhD/git/nnlearn/rys_nuc_project/sib_nuc_position_sequences.fa" : sib_seq_fasta = "F:\\PhD\\git\\nnlearn\\rys_nuc_project\\sib_nuc_position_sequences.fa"

Sys.islinux() ? danio_genome_path = "/media/main/Bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna" : danio_genome_path = "F:\\PhD\\seq\\GRCz11\\GCA_000002035.4_GRCz11_genomic.fna"
Sys.islinux() ? danio_gen_index_path = "/media/main/Bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna.fai" : danio_gen_index_path = "F:\\PhD\\seq\\GRCz11\\GCA_000002035.4_GRCz11_genomic.fna.fai"


@info "Loading master libraries..."
using Distributed, Serialization, ProgressMeter

#JOB CONSTANTS
const BGHMM_lhs= RemoteChannel(()->Channel{Tuple}(30)) #channel to take partitioned BGHMM subsequence likelihoods from
const position_size = 141
const ensemble_size = 300
const no_sources = 100
const source_min_bases = 3
const source_max_bases = position_size
const mixing_prior = 0.15

# #DISTRIBUTED CLUSTER CONSTANTS
# const remote_machine = "10.0.0.12"
const no_local_processes = 1
const no_remote_processes = 0
# #SETUP DISTRIBUTED METAMOTIF LEARNERS
@info "Spawning workers..."
addprocs(no_local_processes, topology=:master_worker)
# #addprocs([(remote_machine,no_remote_processes)], tunnel=true, topology=:master_worker)
pool_size = no_remote_processes + no_local_processes
worker_pool = [i for i in 2:pool_size+1]

@info "Loading worker libraries everywhere..."
@everywhere using BGHMM, BioSequences, nnlearn, Revise

@info "Constructing position dataframe from file at $sib_seq_fasta..."
sib_position_df = nnlearn.make_padded_df(sib_seq_fasta, danio_gff_path, danio_genome_path, danio_gen_index_path, source_max_bases-1)
#sib_position_df = sib_position_df[1:100, :]

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

# @info "Initialising ICA PWM model ensemble for nested sampling..."
# ensemble = nnlearn.init_model_ensemble(ensemble_size, no_sources,mixing_prior,size(coded_obs_set)[2],uninformative_range=source_size_range)
