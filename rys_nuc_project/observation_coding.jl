sib_seq_fasta = "/media/main/Bench/PhD/git/nnlearn/rys_nuc_project/sib_nuc_position_sequences.fa"
position_df_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_positions"
code_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/coded_obs_set"

using Serialization,nnlearn
@info "Constructing position dataframe from file at $sib_seq_fasta..."
sib_position_df = nnlearn.make_position_df(sib_seq_fasta)

@info "Saving position dataframe binary..."
serialize(position_df_binary,sib_position_df)

@info "Setting up observations..."
obs_matrix = nnlearn.observation_setup(sib_position_df)

@info "Serializing coded observation set and sequence offsets..."
serialize(code_binary,(obs_matrix,offsets))