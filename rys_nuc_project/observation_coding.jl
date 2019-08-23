Sys.islinux() ? position_df_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_positions" : position_df_binary = "F:\\PhD\\NGS_binaries\\nnlearn\\BGHMM_sib_positions"
Sys.islinux() ? code_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/coded_obs_set" : code_binary = "F:\\PhD\\NGS_binaries\\nnlearn\\coded_obs_set"

using Serialization,nnlearn

@info "Loading position binary..."
sib_position_df = deserialize(position_df_binary)

@info "Setting up observations..."
coded_seqs, offsets = nnlearn.observation_setup(sib_position_df)

@info "Serializing coded observation set and sequence offsets..."
serialize(code_binary,(coded_seqs,offsets))