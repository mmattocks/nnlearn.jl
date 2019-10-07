#JOB FILEPATHS
danio_gff_path = "/bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3"
danio_genome_path = "/bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna"
danio_gen_index_path = "/bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna.fai"

selected_hmm_output = "/bench/PhD/NGS_binaries/BGHMM/selected_BGHMMs"

position_df_binary = "/bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_positions"

matrix_output = "/bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix"

@info "Loading master libraries..."
using BGHMM, nnlearn, Serialization, ProgressMeter, Distributions

@info "Loading position dataframe..."
sib_position_df = deserialize(position_df_binary)

@info "Masking positions by genome partition and strand, then formatting observations..."
BGHMM.add_partition_masks!(sib_position_df, danio_gff_path)

@info "Setting up for BGHMM likelihood calculations..."
BGHMM_dict = deserialize(selected_hmm_output)

@info "Performing calculations..."
BGHMM_lh_matrix = BGHMM.BGHMM_likelihood_calc(sib_position_df, BGHMM_dict)

@info "Serializing matrix to $matrix_output and position dataframe to $position_df_binary..."
serialize(position_df_binary, sib_position_df)
serialize(matrix_output, BGHMM_lh_matrix)
@info "Job done."