@info "Setting up for job..."
#JOB FILEPATHS
sib_wms_path = "/bench/PhD/NGS_binaries/nnlearn/sib_nuc_position_sequences.fa_wms.tr"
rys_wms_path = "/bench/PhD/NGS_binaries/nnlearn/rys_nuc_position_sequences.fa_wms.tr"

sib_df_binary = "/bench/PhD/NGS_binaries/nnlearn/sib_diff_positions"
rys_df_binary = "/bench/PhD/NGS_binaries/nnlearn/rys_diff_positions"
combined_df_binary = "/bench/PhD/NGS_binaries/nnlearn/combined_diff_positions"

sib_code_binary = "/bench/PhD/NGS_binaries/nnlearn/sib_diff_codes"
rys_code_binary = "/bench/PhD/NGS_binaries/nnlearn/rys_diff_codes"
combined_code_binary = "/bench/PhD/NGS_binaries/nnlearn/combined_diff_codes"

sib_diff_bg = "/bench/PhD/NGS_binaries/nnlearn/sib_diff_bg"
rys_diff_bg = "/bench/PhD/NGS_binaries/nnlearn/rys_diff_bg"
combined_diff_bg = "/bench/PhD/NGS_binaries/nnlearn/combined_diff_bg"

#code_binary = "/bench/PhD/NGS_binaries/nnlearn/coded_obs_set"
#matrix_output = "/bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix"
sib_ensemble = "/bench/PhD/NGS_binaries/nnlearn/sib_ensemble"
rys_ensemble = "/bench/PhD/NGS_binaries/nnlearn/rys_ensemble"
combined_ensemble = "/bench/PhD/NGS_binaries/nnlearn/combined_ensemble"

#JOB CONSTANTS
const position_size = 141
const ensemble_size = 1000
const no_sources = 40
const source_min_bases = 3
const source_max_bases = Int(floor(position_size/2))
@assert source_min_bases < source_max_bases
const source_length_range= source_min_bases:source_max_bases
const mixing_prior = .07
@assert mixing_prior >= 0 && mixing_prior <= 1
const models_to_permute = ensemble_size * 5

const permute_params = [
    [
        ("PSFM",(no_sources, .2, .3)),
        ("PSFM",(no_sources, .8, 1.)),
        ("FM",()),
        ("random",(no_sources)),
        ("reinit",(no_sources))
    ], #worker 2
    [
        ("PSFM",(no_sources, .2, .3)),
        ("PSFM",(no_sources, .8, 1.)),
        ("FM",()),
        ("random",(no_sources)),
        ("reinit",(no_sources))
    ], #worker 3
    [
        ("PSFM",(no_sources, .2, .3)),
        ("FM",()),
        ("merge",(no_sources)),
        ("merge",(no_sources)),
        ("merge",(no_sources)),
        ("random",(no_sources)),
        ("reinit",(no_sources))
    ], #worker 7
    [
        ("PSFM",(no_sources, .8, 1.)),
        ("FM",()),
        ("merge",(no_sources)),
        ("merge",(no_sources)),
        ("merge",(no_sources)),
        ("random",(no_sources)),
        ("reinit",(no_sources))
    ] #worker 8
]
worker_instruction_rand=[true,true,true,true]

const prior_wt=3.0

@info "Loading master libraries..."
using Distributed, Serialization

@info "Adding librarians and workers..."
remote_machine = "10.0.0.3"
remote_pool=addprocs([(remote_machine, 2)], tunnel=true)
librarians=addprocs(2)
local_pool=addprocs(2)
worker_pool=vcat(remote_pool,local_pool)

@info "Loading libraries everywhere..."
@everywhere using nnlearn, Random
Random.seed!(786)

@info "Loading BGHMM likelihood matrix binaries..."
sib_matrix=deserialize(sib_diff_bg)
rys_matrix=deserialize(rys_diff_bg)
combined_matrix=deserialize(combined_diff_bg)

@info "Loading coded observation sets..."
sib_obs = deserialize(sib_code_binary)
rys_obs = deserialize(rys_code_binary)
combined_obs = deserialize(combined_code_binary)

@info "Loading source priors..."
sib_wms = nnlearn.read_fa_wms_tr(sib_wms_path)
rys_wms = nnlearn.read_fa_wms_tr(rys_wms_path)
CartesianIndices
@info "Assembling source priors..."
sib_mix_prior = nnlearn.cluster_mix_prior!(deserialize(sib_df_binary), sib_wms)
rys_mix_prior = nnlearn.cluster_mix_prior!(deserialize(rys_df_binary), rys_wms)
so,ss=size(sib_mix_prior)
ro,rs=size(rys_mix_prior)
combined_mix_prior = falses(so+ro,ss+rs)
combined_mix_prior[1:so,1:ss] = sib_mix_prior
combined_mix_prior[so+1:so+ro,ss+1:ss+rs] = rys_mix_prior

sib_source_priors = nnlearn.assemble_source_priors(no_sources, sib_wms, prior_wt, source_length_range)
rys_source_priors = nnlearn.assemble_source_priors(no_sources, rys_wms, prior_wt, source_length_range)
combined_source_priors = deepcopy(sib_source_priors)
combined_source_priors[ss+1:ss+rs]=rys_source_priors[1:rs]

@info "Initialising sib ICA PWM model ensemble for nested sampling..."
isfile(string(sib_ensemble,'/',"ens")) ? (sib_e = deserialize(string(sib_ensemble,'/',"ens"))) :
    (sib_e = nnlearn.Bayes_IPM_ensemble(worker_pool, sib_ensemble, ensemble_size, sib_source_priors, (sib_mix_prior, mixing_prior), sib_matrix, sib_obs, source_length_range))

@info "Learning differential sib motifs by nested sampling of posterior..."
nnlearn.ns_converge!(sib_e, permute_params, models_to_permute, librarians, worker_pool, model_display=8, backup=(true,5), wkrand=worker_instruction_rand)
serialize(string(sib_ensemble,'/',"ens"), sib_e)

@info "Initialising rys ICA PWM model ensemble for nested sampling..."
isfile(string(rys_ensemble,'/',"ens")) ? (rys_e = deserialize(string(rys_ensemble,'/',"ens"))) :
    (rys_e = nnlearn.Bayes_IPM_ensemble(worker_pool, rys_ensemble, ensemble_size, rys_source_priors, (rys_mix_prior, mixing_prior), rys_matrix, rys_obs, source_length_range))

@info "Learning differential rys motifs by nested sampling of posterior..."
nnlearn.ns_converge!(rys_e, permute_params, models_to_permute, librarians, worker_pool, model_display=8, backup=(true,5), wkrand=worker_instruction_rand)
serialize(string(rys_ensemble,'/',"ens"), rys_e)

@info "Initialising combined ICA PWM model ensemble for nested sampling..."
isfile(string(combined_ensemble,'/',"ens")) ? (combined_e = deserialize(string(combined_ensemble,'/',"ens"))) :
    (combined_e = nnlearn.Bayes_IPM_ensemble(worker_pool, combined_ensemble, ensemble_size, combined_source_priors, (combined_mix_prior, mixing_prior), combined_matrix, combined_obs, source_length_range))

@info "Learning combined motifs by nested sampling of posterior..."
nnlearn.ns_converge!(combined_e, permute_params, models_to_permute, librarians, worker_pool, model_display=8, backup=(true,5), wkrand=worker_instruction_rand)
serialize(string(combined_ensemble,'/',"ens"), combined_e)

rm(worker_pool)

@info "Job done!"