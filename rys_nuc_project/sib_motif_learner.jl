#JOB FILEPATHS
Sys.islinux() ? danio_gff_path = "/media/main/Bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3" : danio_gff_path = "F:\\PhD\\seq\\GRCz11\\Danio_rerio.GRCz11.94.gff3"
Sys.islinux() ? selected_hmm_output = "/media/main/Bench/PhD/NGS_binaries/BGHMM/selected_BGHMMs" : selected_hmm_output = "F:\\PhD\\NGS_binaries\\BGHMM\\selected_BGHMMs"
Sys.islinux() ? sib_seq_fasta = "/media/main/Bench/PhD/git/nnlearn/rys_nuc_project/sib_nuc_position_sequences.fa" : sib_seq_fasta = "F:\\PhD\\git\\nnlearn\\rys_nuc_project\\sib_nuc_position_sequences.fa"

@info "Loading master libraries..."
using Distributed, Serialization, ProgressMeter

#JOB CONSTANTS
const BGHMM_lhs= RemoteChannel(()->Channel{Tuple}(30)) #channel to take partitioned BGHMM subsequence likelihoods from
const ensemble_size = 300
const no_sources = 30
const source_size_range = 3:50
const mixing_prior = 0.15

#DISTRIBUTED CLUSTER CONSTANTS
const remote_machine = "10.0.0.12"
const no_local_processes = 8
const no_remote_processes = 0
#SETUP DISTRIBUTED BAUM WELCH LEARNERS
@info "Spawning workers..."
addprocs(no_local_processes, topology=:master_worker)
#addprocs([(remote_machine,no_remote_processes)], tunnel=true, topology=:master_worker)
pool_size = no_remote_processes + no_local_processes
worker_pool = [i for i in 2:pool_size+1]

@info "Loading worker libraries everywhere..."
@everywhere using BGHMM, BioSequences, nnlearn

@info "Constructing position dataframe from file at $sib_seq_fasta..."
sib_position_df = nnlearn.make_position_df(BioSequences.FASTA.Reader(open(sib_seq_fasta,"r")))
sib_position_df = sib_position_df[1:100, :]

@info "Masking positions by genome partition and formatting observations..."
masked_position_df = BGHMM.add_partition_masks(sib_position_df, danio_gff_path)
coded_obs_set, BGHMM_mask = nnlearn.observation_setup(masked_position_df)

@info "Setting up for BGHMM likelihood calculations..."
BGHMM_dict = deserialize(selected_hmm_output)
BGHMM_likelihood_queue, jobcount = nnlearn.queue_BGHMM_likelihood_calcs(coded_obs_set, BGHMM_mask)

@info "Distributing BGHMM likelihood jobs to workers..."
for worker in worker_pool
    remote_do(nnlearn.process_BGHMM_likelihood_queue!, worker, BGHMM_likelihood_queue, BGHMM_lhs, BGHMM_dict)
end

BGHMM_meter=ProgressMeter.Progress(jobcount,"Overall batch progress:")
BGHMM_lh_matrix = zeros(size(coded_obs_set))
jobcounter=jobcount
while jobcounter > 0
    wait(BGHMM_lhs)
    subseq_lh, start_index = take!(BGHMM_lhs)
    end_index = CartesianIndex(start_index[1] + length(subseq_lh) - 1, start_index[2])
    BGHMM_lh_matrix[start_index:end_index] = subseq_lh
    global jobcounter-=1

    ProgressMeter.update!(BGHMM_meter, jobcount-jobcounter)
end

@info "Initialising ICA PWM model ensemble for nested sampling..."
ensemble = nnlearn.init_model_ensemble(ensemble_size, no_sources,mixing_prior,size(coded_obs_set)[2],uninformative_range=source_size_range)
