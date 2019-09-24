#JOB FILEPATHS
Sys.islinux() ? code_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/coded_obs_set" : code_binary = "F:\\PhD\\NGS_binaries\\nnlearn\\coded_obs_set"
Sys.islinux() ? matrix_output = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix" : matrix_output = "F:\\PhD\\NGS_binaries\\nnlearn\\BGHMM_sib_matrix"
Sys.islinux() ? ensemble_directory = "/media/main/Bench/PhD/NGS_binaries/nnlearn/sib_ensemble/" : ensemble_directory = "F:\\PhD\\NGS_binaries\\nnlearn\\sib_ensemble\\"
!ispath(ensemble_directory) && mkpath(ensemble_directory)
Sys.islinux() ? converged_sample = "/media/main/Bench/PhD/NGS_binaries/nnlearn/converged_sample" : converged_sample = "F:\\PhD\\NGS_binaries\\nnlearn\\converged_sample"

@info "Loading libraries..."
using Distributions, nnlearn, ProgressMeter, Serialization

#JOB CONSTANTS
const position_size = 141
const ensemble_size = 100
const no_sources = 30
const source_min_bases = 3
const source_max_bases = Int(round(position_size/2))
@assert source_min_bases < source_max_bases
const source_length_range = source_min_bases:source_max_bases
const mixing_prior = no_sources/1000
@assert mixing_prior >= 0 && mixing_prior <= 1

#1.34gb
@info "Loading BGHMM likelihood matrix binary..."
BGHMM_lh_matrix = deserialize(matrix_output)
#4.75GB
@info "Loading coded observation set and offsets..."
coded_seqs = deserialize(code_binary)
#8.30GB
@info "Assembling source priors..."
source_priors = nnlearn.assemble_source_priors(no_sources, [], source_length_range)
#8.31GB
@info "Initialising ICA PWM model ensemble for nested sampling..."
ensemble = nnlearn.Bayes_IPM_ensemble(ensemble_directory, ensemble_size, source_priors, mixing_prior, BGHMM_lh_matrix, coded_seqs, position_size, source_length_range)

ultralight_even_perm = (10, ones(3)/3, Uniform(.00001,.01))
light_even_perm = (300, ones(3)/3, Uniform(.00001,.01))
med_even_perm = (900, ones(3)/3, Uniform(.0001,.03))
heavy_even_perm = (3000, ones(3)/3, Uniform(.001,.05))

permute_params = [(ultralight_even_perm,3), (light_even_perm, 3), (med_even_perm, 3), (heavy_even_perm, 3)]
models_to_permute = ensemble_size

serialize(converged_sample, nnlearn.nested_sample_posterior_to_convergence!(ensemble, permute_params, models_to_permute))

@info "Job done!"