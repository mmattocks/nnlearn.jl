#JOB FILEPATHS
Sys.islinux() ? danio_gff_path = "/media/main/Bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3" : danio_gff_path = "F:\\PhD\\seq\\GRCz11\\Danio_rerio.GRCz11.94.gff3"
Sys.islinux() ? selected_hmm_output = "/media/main/Bench/PhD/NGS_binaries/BGHMM/selected_BGHMMs" : selected_hmm_output = "F:\\PhD\\NGS_binaries\\BGHMM\\selected_BGHMMs"
Sys.islinux() ? sib_seq_fasta = "/media/main/Bench/PhD/git/nnlearn/rys_nuc_project/sib_nuc_position_sequences.fa" : sib_seq_fasta = "F:\\PhD\\git\\nnlearn\\rys_nuc_project\\sib_nuc_position_sequences.fa"

@info "Loading libraries..."
using BGHMM, BioSequences, nnlearn, Serialization

@info "Constructing position dataframe from file at $sib_seq_fasta..."
sib_position_df = nnlearn.make_position_df(BioSequences.FASTA.Reader(open(sib_seq_fasta,"r")))
sib_position_df = sib_position_df[1:100, :]

@info "Masking positions by genome partition and formatting observations..."
masked_position_df = BGHMM.add_partition_masks(sib_position_df, danio_gff_path)
coded_obs_set, BGHMM_mask = nnlearn.observation_setup(masked_position_df)

@info "Setting up for BGHMM likelihood calculations..."
BGHMM_dict = deserialize(selected_hmm_output)
BGHMM_likelihood_queue = nnlearn.queue_BGHMM_likelihood_calcs(coded_obs_set, BGHMM_mask, BGHMM_dict)
