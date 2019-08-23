#JOB FILEPATHS
Sys.islinux() ? danio_gff_path = "/media/main/Bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3" : danio_gff_path = "F:\\PhD\\seq\\GRCz11\\Danio_rerio.GRCz11.94.gff3"
Sys.islinux() ? selected_hmm_output = "/media/main/Bench/PhD/NGS_binaries/BGHMM/selected_BGHMMs" : selected_hmm_output = "F:\\PhD\\NGS_binaries\\BGHMM\\selected_BGHMMs"
Sys.islinux() ? sib_seq_fasta = "/media/main/Bench/PhD/git/nnlearn/rys_nuc_project/sib_nuc_position_sequences.fa" : sib_seq_fasta = "F:\\PhD\\git\\nnlearn\\rys_nuc_project\\sib_nuc_position_sequences.fa"

Sys.islinux() ? danio_genome_path = "/media/main/Bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna" : danio_genome_path = "F:\\PhD\\seq\\GRCz11\\GCA_000002035.4_GRCz11_genomic.fna"
Sys.islinux() ? danio_gen_index_path = "/media/main/Bench/PhD/seq/GRCz11/GCA_000002035.4_GRCz11_genomic.fna.fai" : danio_gen_index_path = "F:\\PhD\\seq\\GRCz11\\GCA_000002035.4_GRCz11_genomic.fna.fai"

Sys.islinux() ? position_df_binary = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_positions" : position_df_binary = "F:\\PhD\\NGS_binaries\\nnlearn\\BGHMM_sib_positions"
Sys.islinux() ? matrix_output = "/media/main/Bench/PhD/NGS_binaries/nnlearn/BGHMM_sib_matrix" : matrix_output = "F:\\PhD\\NGS_binaries\\nnlearn\\BGHMM_sib_matrix"

@info "Loading master libraries..."
using BGHMM, nnlearn, Serialization, ProgressMeter, Distributions

#JOB CONSTANTS
const pad = 141 #number of bases to take 5' of the position, source length limited to <this number to produce scores for all bases in a position

@info "Constructing position dataframe from file at $sib_seq_fasta..."
sib_position_df = BGHMM.make_padded_df(sib_seq_fasta, danio_gff_path, danio_genome_path, danio_gen_index_path, pad)

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