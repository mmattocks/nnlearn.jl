#JOB FILEPATHS
prior_wms_path = "/bench/PhD/NGS_binaries/nnlearn/sib_nuc_position_sequences.fa_wms.tr"
code_binary = "/bench/PhD/NGS_binaries/nnlearn/coded_obs_set"
matrix_output = "/bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix"
ensemble_directory = "/bench/PhD/NGS_binaries/nnlearn/sib_ensemble/"
!ispath(ensemble_directory) && mkpath(ensemble_directory)
converged_sample = "/media/main/Bench/PhD/NGS_binaries/nnlearn/converged_sample"

#JOB CONSTANTS
const position_size = 141
const ensemble_size = 5000
const no_sources = 50
const source_min_bases = 3
const source_max_bases = position_size
@assert source_min_bases < source_max_bases
const source_length_range = source_min_bases:source_max_bases
const mixing_prior = .1
@assert mixing_prior >= 0 && mixing_prior <= 1
const models_to_permute = ensemble_size * 5
const permute_params = [("permute",(50,100000)),("permute",(1000,100)),("merge",(no_sources*3)),("init",(100))]

using Distributed, Serialization

@info "Adding librarians and workers..."
remote_machine = "10.0.0.2"
librarians=addprocs(1)
local_pool=addprocs(2)
remote_pool=addprocs(remote_machine, 1)
worker_pool=vcat(local_pool,remote_pool)

@info "Loading libraries..."
@everywhere using nnlearn, Random
@everywhere Random.seed!(786)

@info "Loading BGHMM likelihood matrix binary..."
BGHMM_lh_matrix = deserialize(matrix_output)

@info "Loading coded observation set and offsets..."
coded_seqs = deserialize(code_binary)

@info "Loading source priors..."
wm_priors = nnlearn.read_fa_wms_tr(prior_wms_path)

@info "Assembling source priors..."
source_priors = nnlearn.assemble_source_priors(no_sources, wm_priors, source_length_range)

@info "Initialising ICA PWM model ensemble for nested sampling..."
ensemble = nnlearn.Bayes_IPM_ensemble(ensemble_directory, ensemble_size, source_priors, mixing_prior, BGHMM_lh_matrix, coded_seqs, source_length_range)

serialize(converged_sample, nnlearn.ns_converge!(ensemble, permute_params, models_to_permute, librarians, worker_pool))

@info "Job done!"