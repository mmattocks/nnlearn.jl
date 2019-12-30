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
const ensemble_size = 500
const no_sources = 8
const source_min_bases = 3
const source_max_bases = 10
@assert source_min_bases < source_max_bases
const source_length_range= source_min_bases:source_max_bases
const mixing_prior = .07
@assert mixing_prior >= 0 && mixing_prior <= 1
const models_to_permute = ensemble_size * 5

local_job_sets=[
([
    ("PS", (no_sources)),
    ("PM", (no_sources)),
    ("PSFM", (no_sources)),
    ("PSFM", (no_sources*10, .8, 1.)),
    ("FM", ()),
    ("DM", (no_sources)),
    ("SM", (no_sources)),
    ("RD", (no_sources)),
    ("RI", (no_sources)),
    ("EM", (no_sources))
],[.025, .025, .025, .025, .775, .025, .025, .025, .025, .025]),
([
    ("PS", (no_sources)),
    ("PM", (no_sources)),
    ("PSFM", (no_sources)),
    ("PSFM", (no_sources, 0., 1.)),
    ("PSFM", (no_sources, 0.8, .0)),
    ("FM", ()),
    ("DM", (no_sources)),
    ("SM", (no_sources)),
    ("RD", (no_sources)),
    ("RI", (no_sources)),
    ("EM", (no_sources))
],[.15, .05, .05, .025, .025, .05, 0.15, 0.15, 0.15, 0.05, .15]),
([
    ("PS", (no_sources)),
    ("PM", (no_sources)),
    ("PSFM", (no_sources)),
    ("PSFM", (no_sources, 0., 1.)),
    ("PSFM", (no_sources, 0.8, .0)),
    ("FM", ()),
    ("DM", (no_sources)),
    ("SM", (no_sources)),
    ("RD", (no_sources)),
    ("RI", (no_sources)),
    ("EM", (no_sources))
],[.15, .05, .05, .025, .025, .05, 0.15, 0.15, 0.15, 0.05, .15]),
# ([
#     ("PS", (no_sources)),
#     ("PM", (no_sources)),
#     ("PSFM", (no_sources,.2,.05)),
#     ("DM", (no_sources)),
#     ("SM", (no_sources)),
#     ("RD", (no_sources)),
#     ("EM", (no_sources))
# ],[.10, .15, .15, 0.15, 0.15, 0.15, .15])
]
remote_job_sets=[
([
    ("PS", (no_sources)),
    ("PM", (no_sources)),
    ("PSFM", (no_sources)),
    ("PSFM", (no_sources*10, .8, 1.)),
    ("FM", ()),
    ("RD", (no_sources)),
    ("RI", (no_sources)),
    ("EM", (no_sources))
],[.025, .025, .025, .025, .825, .025, .025, .025]),
([
    ("PS", (no_sources)),
    ("PM", (no_sources)),
    ("PSFM", (no_sources)),
    ("PSFM", (no_sources, 0., 1.)),
    ("PSFM", (no_sources, 0.8, .0)),
    ("FM", ()),
    ("RD", (no_sources)),
    ("RI", (no_sources)),
    ("EM", (no_sources))
],[.20, .10, .10, .05, .05, .10, 0.15, 0.10, .15]),
([
    ("PS", (no_sources)),
    ("PM", (no_sources)),
    ("PSFM", (no_sources)),
    ("PSFM", (no_sources, 0., 1.)),
    ("PSFM", (no_sources, 0.8, .0)),
    ("FM", ()),
    ("RD", (no_sources)),
    ("RI", (no_sources)),
    ("EM", (no_sources))
],[.20, .10, .10, .05, .05, .10, 0.15, 0.10, .15]),
# ([
#     ("PS", (no_sources)),
#     ("PM", (no_sources)),
#     ("PSFM", (no_sources,.2,.05)),
#     ("RD", (no_sources)),
#     ("EM", (no_sources))
# ],[.20, .25, .25, 0.15, .15])
]
job_limit=6
const prior_wt=1.2

@info "Loading master libraries..."
using Distributed, Serialization

@info "Adding librarians and workers..."
no_local_procs=2
no_remote_procs=2
remote_machine = "10.0.0.3"
remote_pool=addprocs([(remote_machine, no_remote_procs)], tunnel=true, topology=:master_worker)
local_pool=addprocs(no_local_procs, topology=:master_worker)
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

@info "Loading informative source priors..."
sib_wms = nnlearn.read_fa_wms_tr(sib_wms_path)
rys_wms = nnlearn.read_fa_wms_tr(rys_wms_path)
sib_mix_prior = nnlearn.cluster_mix_prior!(deserialize(sib_df_binary), sib_wms)
rys_mix_prior = nnlearn.cluster_mix_prior!(deserialize(rys_df_binary), rys_wms)

@info "Filtering informative priors..."
sib_prior_wms = nnlearn.filter_priors(Int(floor(no_sources/2)), source_max_bases, sib_wms, sib_mix_prior)
rys_prior_wms = nnlearn.filter_priors(Int(floor(no_sources/2)), source_max_bases, rys_wms, rys_mix_prior)
combined_prior_wms = nnlearn.combine_filter_priors(Int(floor(no_sources/2)), source_max_bases, (sib_wms, rys_wms), (sib_mix_prior, rys_mix_prior))

@info "Assembling source priors..."
sib_source_priors = nnlearn.assemble_source_priors(no_sources, sib_prior_wms, prior_wt, source_length_range)
rys_source_priors = nnlearn.assemble_source_priors(no_sources, rys_prior_wms, prior_wt, source_length_range)
combined_source_priors = nnlearn.assemble_source_priors(no_sources, combined_prior_wms, prior_wt, source_length_range)

@info "Initialising sib ICA PWM model ensemble for nested sampling..."
isfile(string(sib_ensemble,'/',"ens")) ? (sib_e = deserialize(string(sib_ensemble,'/',"ens"))) :
    (sib_e = nnlearn.Bayes_IPM_ensemble(worker_pool, sib_ensemble, ensemble_size, sib_source_priors, (falses(0,0), mixing_prior), sib_matrix, sib_obs, source_length_range))

job_set_thresh=[-Inf,sib_e.naive_lh,sib_e.naive_lh+230000]
remote_param_sets=[(remote_job_sets,job_set_thresh,job_limit) for i in 1:no_remote_procs]
local_param_sets=[(local_job_sets,job_set_thresh,job_limit) for i in 1:no_local_procs]
permute_params=vcat(remote_param_sets,local_param_sets)

@info "Learning differential sib motifs by nested sampling of posterior..."
nnlearn.ns_converge!(sib_e, permute_params, models_to_permute, [1], worker_pool, model_display=8, backup=(true,5))
serialize(string(sib_ensemble,'/',"ens"), sib_e)

@info "Initialising rys ICA PWM model ensemble for nested sampling..."
isfile(string(rys_ensemble,'/',"ens")) ? (rys_e = deserialize(string(rys_ensemble,'/',"ens"))) :
    (rys_e = nnlearn.Bayes_IPM_ensemble(worker_pool, rys_ensemble, ensemble_size, rys_source_priors, (falses(0,0), mixing_prior), rys_matrix, rys_obs, source_length_range))

job_set_thresh=[-Inf,rys_e.naive_lh,rys_e.naive_lh+230000]
remote_param_sets=[(remote_job_sets,job_set_thresh,job_limit) for i in 1:no_remote_procs]
local_param_sets=[(local_job_sets,job_set_thresh,job_limit) for i in 1:no_local_procs]
permute_params=vcat(remote_param_sets,local_param_sets)

@info "Learning differential rys motifs by nested sampling of posterior..."
nnlearn.ns_converge!(rys_e, permute_params, models_to_permute, [1], worker_pool, model_display=8, backup=(true,5))
serialize(string(rys_ensemble,'/',"ens"), rys_e)

@info "Initialising combined ICA PWM model ensemble for nested sampling..."
isfile(string(combined_ensemble,'/',"ens")) ? (combined_e = deserialize(string(combined_ensemble,'/',"ens"))) :
    (combined_e = nnlearn.Bayes_IPM_ensemble(worker_pool, combined_ensemble, ensemble_size, combined_source_priors, (falses(0,0), mixing_prior), combined_matrix, combined_obs, source_length_range))

job_set_thresh=[-Inf,combined_e.naive_lh,combined_e.naive_lh+230000]
remote_param_sets=[(remote_job_sets,job_set_thresh,job_limit) for i in 1:no_remote_procs]
local_param_sets=[(local_job_sets,job_set_thresh,job_limit) for i in 1:no_local_procs]
permute_params=vcat(remote_param_sets,local_param_sets)
    
@info "Learning combined motifs by nested sampling of posterior..."
nnlearn.ns_converge!(combined_e, permute_params, models_to_permute, [1], worker_pool, model_display=8, backup=(true,5))
serialize(string(combined_ensemble,'/',"ens"), combined_e)

rm(worker_pool)

@info "Job done!"