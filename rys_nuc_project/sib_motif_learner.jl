@info "Setting up for job..."
#JOB FILEPATHS
prior_wms_path = "/bench/PhD/NGS_binaries/nnlearn/sib_nuc_position_sequences.fa_wms.tr"
code_binary = "/bench/PhD/NGS_binaries/nnlearn/coded_obs_set"
matrix_output = "/bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix"
ensemble_directory = "/bench/PhD/NGS_binaries/nnlearn/sib_ensemble/"
!ispath(ensemble_directory) && mkpath(ensemble_directory)
converged_sample = "/media/main/Bench/PhD/NGS_binaries/nnlearn/converged_sample"

#JOB CONSTANTS
const position_size = 141
const ensemble_size = 1000
const no_sources = 50
const source_min_bases = 3
const source_max_bases = Int(ceil(position_size/2))
@assert source_min_bases < source_max_bases
const source_length_range = source_min_bases:source_max_bases
const mixing_prior = .1
@assert mixing_prior >= 0 && mixing_prior <= 1
const models_to_permute = ensemble_size * 5
const permute_params = [("permute",(100,100)),("permute",(10,5000,[.8,.1,.1])),("merge",(no_sources*3)),("init",(100))]
const prior_wt=3.0

using Distributed, Serialization

@info "Adding librarians and workers..."
remote_machine = "10.0.0.2"
remote_pool=addprocs([(remote_machine, 1)], tunnel=true)
librarians=addprocs(2)
local_pool=addprocs(2)
worker_pool=vcat(remote_pool,local_pool)
# worker_pool=local_pool

@info "Loading libraries..."
# using nnlearn, Random, Serialization
@everywhere using nnlearn, Random
Random.seed!(786)

@info "Loading BGHMM likelihood matrix binary..."
BGHMM_lh_matrix = deserialize(matrix_output)

@info "Loading coded observation set and offsets..."
coded_seqs = deserialize(code_binary)

@info "Loading source priors..."
wm_priors = nnlearn.read_fa_wms_tr(prior_wms_path)

@info "Assembling source priors..."
source_priors = nnlearn.assemble_source_priors(no_sources, wm_priors, prior_wt, source_length_range)

@info "Initialising ICA PWM model ensemble for nested sampling..."
isfile(string(ensemble_directory,'/',"ens")) ? (ensemble = deserialize(string(ensemble_directory,'/',"ens"))) :
    (ensemble = nnlearn.Bayes_IPM_ensemble(ensemble_directory, ensemble_size, source_priors, mixing_prior, BGHMM_lh_matrix, coded_seqs, source_length_range))

@info "Learning motifs by nested sampling of posterior..."
serialize(converged_sample, nnlearn.ns_converge!(ensemble, permute_params, models_to_permute, librarians, worker_pool, backup=(true,5)))
# serialize(converged_sample, nnlearn.ns_converge!(ensemble, permute_params, models_to_permute, backup=(true,5)))

@info "Job done!"