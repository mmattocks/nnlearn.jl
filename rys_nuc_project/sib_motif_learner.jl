#JOB FILEPATHS
Sys.islinux() ? danio_gff_path = "/media/main/Bench/PhD/seq/GRCz11/Danio_rerio.GRCz11.94.gff3" : danio_gff_path = "F:\\PhD\\seq\\GRCz11\\Danio_rerio.GRCz11.94.gff3"
Sys.islinux() ? selected_hmm_output = "/media/main/Bench/PhD/NGS_binaries/BGHMM/selected_BGHMMs" : selected_hmm_output = "F:\\PhD\\NGS_binaries\\BGHMM\\selected_BGHMMs"
Sys.islinux() ? sib_seq_fasta = "/media/main/Bench/PhD/git/nnlearn/sib_nuc_position_sequences.fa" : sib_seq_fasta = "F:\\PhD\\git\\nnlearn\\sib_nuc_position_sequences.fa"

using BioSequences, nnlearn, Serialization

BGHMM_dict = deserialize(selected_hmm_output)
sib_position_df = nnlearn.make_position_df(BioSequences.FASTA.Reader(open(sib_seq_fasta,"r"))

coded_obs_set = nnlearn.observation_setup(sib_position_df)
