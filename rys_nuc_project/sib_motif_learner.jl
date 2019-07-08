#JOB FILEPATHS
Sys.islinux() ? code_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/coded_obs_set" : code_binary = "F:\\PhD\\NGS_binaries\\nnlearn\\coded_obs_set"
Sys.islinux() ? matrix_output = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix" : matrix_output = "F:\\PhD\\NGS_binaries\\nnlearn\\BGHMM_sib_matrix"
Sys.islinux() ? ensemble_directory = "/media/main/Bench/PhD/NGS_binaries/nnlearn/sib_ensemble/" : ensemble_directory = "F:\\PhD\\NGS_binaries\\nnlearn\\sib_ensemble\\"
!ispath(ensemble_directory) && mkpath(ensemble_directory)
Sys.islinux() ? converged_sample = "/media/main/Bench/PhD/NGS_binaries/nnlearn/converged_sample" : converged_sample = "F:\\PhD\\NGS_binaries\\nnlearn\\converged_sample"

@info "Loading master libraries..."
using Distributed, Serialization, ProgressMeter, Distributions

#JOB CONSTANTS
const BGHMM_lhs= RemoteChannel(()->Channel{Tuple}(30)) #channel to take partitioned BGHMM subsequence likelihoods from
const position_size = 141
const ensemble_size = 100
const no_sources = 10
const source_min_bases = 3
const source_max_bases = position_size
@assert source_min_bases < source_max_bases
const source_length_range = source_min_bases:source_max_bases
const mixing_prior = 0.05
@assert mixing_prior >= 0 && mixing_prior <= 1
const jobs_chan= RemoteChannel(()->Channel{Tuple}(Inf)) #channel to hold scoring/llh jobs
const results_chan= RemoteChannel(()->Channel{Tuple}(Inf)) #channel to take scores off of

# #DISTRIBUTED CLUSTER CONSTANTS
# const remote_machine = "10.0.0.12"
# const no_local_processes = 12
# const no_remote_processes = 0
# #SETUP DISTRIBUTED METAMOTIF LEARNERS
# @info "Spawning workers..."
# addprocs(no_local_processes, topology=:master_worker)
# # addprocs([(remote_machine,no_remote_processes)], tunnel=true, topology=:master_worker)
# pool_size = no_remote_processes + no_local_processes
# worker_pool = [i for i in 2:pool_size+1]
# const dist_params = (true, jobs_chan, results_chan)

@info "Loading worker libraries everywhere..."
@everywhere using nnlearn

# for worker in worker_pool
#     remote_do(nnlearn.IPM_likelihood_worker, worker, jobs_chan, results_chan)
# end

@info "Loading BGHMM likelihood matrix binary..."
BGHMM_lh_matrix = deserialize(matrix_output)

#BGHMM_lh_matrix = BGHMM_lh_matrix[:,1:200]

@info "Loading coded observation set and offsets..."
(coded_seqs, offsets) = deserialize(code_binary)

#coded_seqs = coded_seqs[:,1:200]
#offsets = offsets[1:200]

@info "Assembling source priors..."
source_priors = nnlearn.assemble_source_priors(no_sources, [], source_length_range)

@info "Initialising ICA PWM model ensemble for nested sampling..."
ensemble = nnlearn.Bayes_IPM_ensemble(ensemble_directory, ensemble_size, source_priors, mixing_prior, BGHMM_lh_matrix, coded_seqs, position_size, offsets, source_length_range)

ultralight_even_perm = (10, ones(3)/3, Uniform(.00001,.01))
light_even_perm = (300, ones(3)/3, Uniform(.00001,.01))
med_even_perm = (900, ones(3)/3, Uniform(.0001,.03))
heavy_even_perm = (3000, ones(3)/3, Uniform(.001,.05))

permute_params = [(ultralight_even_perm,3), (light_even_perm, 3), (med_even_perm, 3), (heavy_even_perm, 3)]
models_to_permute = ensemble_size

serialize(converged_sample, nnlearn.nested_sample_posterior_to_convergence!(ensemble, permute_params, models_to_permute))

@info "Job done!"