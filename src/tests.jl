function get_fake_BGHMM_dict()
    exon_hmm = MS_HMMBase.HMM([.5, .5], [.9 .1; .2 .8], [Categorical([.4,.1,.1,.4]), Categorical([.1, .4, .4, .1])])
    periexonic_hmm = MS_HMMBase.HMM([.8, .2], [.99 .01; .1 .9], [Categorical([.4,.1,.1,.4]), Categorical([.1, .4, .4, .1])])
    intergenic_hmm = MS_HMMBase.HMM([.5, .5], [.99 .01; .99 .01], [Categorical([.3,.2,.2,.3]), Categorical([.2, .3, .3, .2])])
    return Dict{String,Tuple{HMM,Int64,Float64}}("exon"=>(exon_hmm,0,0.0), "periexonic" => (periexonic_hmm,0,0.0), "intergenic"=>(intergenic_hmm,0,0.0))
end


function verify_BGHMM_matrix_assembly()
    using Distributed, ProgressMeter

    const BGHMM_lhs= RemoteChannel(()->Channel{Tuple}(30)) #channel to take partitioned BGHMM subsequence likelihoods from
    const no_local_processes = 1
    @info "Spawning workers..."
    addprocs(no_local_processes, topology=:master_worker)
    pool_size = no_remote_processes + no_local_processes
    worker_pool = [i for i in 2:pool_size+1]
    @everywhere using BGHMM, BioSequences, nnlearn

    test_dict = get_fake_BGHMM_dict()
    test_df = DataFrame(SeqID=String[], Start=Int64[], End=Int64[], PosSeq=DNASequence[], FwdPad=DNASequence[], RevPad=DNASequence[], MaskMatrix=Matrix[])
    push!(test_df,
        ["1",
        18584931,18585071,
        DNASequence("TACCTACCCTCCCTTCACTCTCAATGGAGAGTATCTCCAGCGCCTCGAGATGAGACAAGCCTTGGTAATCATCAAACAAGATTCCATAGTGTGCGGTTGGTTCTGTGACCATCTGACTGCTCAGTCTCGCTCTCTCCTTCT"),
        DNASequence("TACCTACCCTCCCTTCACTCTCAATGGAGAGTATCTCCAGCGCCTCGAGATGAGACAAGCCTTGGTAATCATCAAACAAGATTCCATAGTGTGCGGTTGGTTCTGTGACCATCTGACTGCTCAGTCTCGCTCTCTCCTTCTCCTTCGCTTCTTTTAACATCTGCAGACAACACAGCAAAAGCACATAGATCAGACACAGTAAGTGCCAGTCTTATCAACATATGCTAAATTATGGCAGGACTTTCCCACAAAACCACAATTTATGCAAGTATCTTAAAGCA"),
        DNASequence("AGAAGGAGAGAGCGAGACTGAGCAGTCAGATGGTCACAGAACCAACCGCACACTATGGAATCTTGTTTGATGATTACCAAGGCTTGTCTCATCTCGAGGCGCTGGAGATACTCTCCATTGAGAGTGAAGGGAGGGTAGGTATTACACACCACCAACATTCACCCACCCATAATATCTTTTACCACCAAGAAACAAGCAAAAAAATGCAGTATTGTCATCAGAAATTAATCTCTAGCCAATATTTTTTCCCCTGTATTTTTGCTCATTATCCAGTAAAATAA"),
        [2 -1; 2 -1; 2 -1; 2 -1; 2 -1; 2 -1; 2 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1; 3 -1]])

        BGHMM_likelihood_queue, jobcount, lh_matrix_size = nnlearn.queue_BGHMM_likelihood_calcs(test_df, test_dict)

        
        @info "Distributing BGHMM likelihood jobs to workers..."
        for worker in worker_pool
            remote_do(nnlearn.process_BGHMM_likelihood_queue!, worker, BGHMM_likelihood_queue, BGHMM_lhs, BGHMM_dict)
        end

        BGHMM_lh_matrix = zeros(lh_matrix_size) #T, Strand, O
        @showprogress 1 "Overall batch progress:" 1 for job in jobcount:-1:1
            @debug "$job lh jobs remaining"
            wait(BGHMM_lhs)
            jobid, frag_lhs = take!(BGHMM_lhs)
            (frag_start, o, partition, strand) = jobid
            if strand == 0
                frag_lhs[:,2]=reverse(frag_lhs[:,2],1)
            elseif strand == -1
                frag_lhs = frag_lhs[end:-1:1,:]
            end
            BGHMM_lh_matrix[frag_start:frag_start+size(frag_lhs)[1]-1,:,o] = frag_lhs
        end

        println(test_df[1,7])
        println(BGHMM_lh_matrix)
        rmprocs(worker_pool)
end