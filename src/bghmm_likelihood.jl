#function to obtain positional likelihoods for a sequence under a given HMM. Mostly derived from MS_HMMBase.jl mle function
function get_BGHMM_symbol_lh(seq::Matrix{Int64}, hmm::HMM)
    symbol_lhs = zeros(length(seq))
    length_mask = [length(seq)-1]
    
    lls =  MS_HMMBase.log_likelihoods(hmm, seq) #obtain log likelihoods for sequences and states
    log_α = MS_HMMBase.messages_forwards_log(hmm.π0, hmm.π, lls, length_mask) #get forward messages
    log_β = MS_HMMBase.messages_backwards_log(hmm.π, lls, length_mask) # get backwards messages

    #calculate observation probability and γ weights
    K,Tmaxplus1,Strand = size(lls) #the last T value is the 0 end marker of the longest T

    #transforms to cut down log_ξ, log_γ assignment times
    lls = permutedims(lls, [2,1,3]) # from (K,T) to (T,K)
    log_α = permutedims(log_α, [2,1,3])
    log_β = permutedims(log_β, [2,1,3])

    log_γ = fill(-Inf, Tmaxplus1,K)
    log_pobs = logsumexp(MS_HMMBase.log_prob_sum.(log_α[1,:], log_β[1,:]))

    for i = 1:K
        for t = 1:Tmaxplus1-2
            log_γ[t,i] = MS_HMMBase.log_prob_sum(log_α[t,i],log_β[t,i],-log_pobs)
        end
        t = Tmaxplus1-1 #log_ξ & log_γ T = 0
        log_γ[t,i] = 0
    end

    for t in 1:size(seq)[1]-2 #iterating down the sequence, stopping short of the final base for which no reasonable estimate is available (we pass along a 0)
        symbol_lh::Float64 = -Inf #ie 0 in logspace
        for k = 1:K #iterate over states
                state_symbol_lh::Float64 = MS_HMMBase.log_prob_sum(log_γ[t,k], log(hmm.D[k].p[seq[t]])) #state symbol likelihood is the γ weight * the state symbol probability (log implementation)
                symbol_lh = logaddexp(symbol_lh, state_symbol_lh) #sum the probabilities over states
        end
        symbol_lhs[t] = symbol_lh
    end

    return symbol_lhs[1:end-1] #remove trailing index position
end

#function to generate substring BGHMM likelihood jobs from an observation set and mask of BGHMM partitions
function queue_BGHMM_likelihood_calcs(observations::DataFrame, BGHMM_dict::Dict{String,Tuple{HMM, Int64, Float64}})
    ##INTENT: DECOMPOSE OBSERVATIONS INTO VECTORS FRAGMENTED BY PARTITION AND STRAND FOR SUBMISSION TO INDIVIDUAL PARTITION HMMs, ALONGSIDE TUPLE OF TXO COORDINATES AND STRAND DIRECTION FOR REASSEMBLY
    BGHMM_fragments = fragment_observations_by_BGHMM(observations.PadSeq, observations.MaskMatrix, observations.SeqOffset)

    lh_job_queue = RemoteChannel(()->Channel{Tuple{Tuple,BioSequence{DNAAlphabet{4}}}}(Inf))
    job_count = 0
    for (jobid, frag) in BGHMM_fragments #iterate over the subseq vector, pushing jobs to the RemoteChannel
        put!(lh_job_queue, (jobid, frag))
        job_count += 1
    end

    return lh_job_queue, job_count, ((findmax(length.(collect(values(observations.PadSeq))))[1]), length(observations.PadSeq))
end

function fragment_observations_by_BGHMM(seqs::Vector{BioSequence{DNAAlphabet{4}}}, masks::Vector{Matrix{Int64}},offsets::Vector{Int64})
    likelihood_jobs = Dict{Tuple,BioSequence{DNAAlphabet{4}}}()
    @showprogress 1 "Fragmenting observations by partition..." for (o, obs_seq) in enumerate(seqs)
        mask = masks[o]
        frags = Vector{BioSequence{DNAAlphabet{4}}}() #container for all subsequences in observation
        frag = BioSequence{DNAAlphabet{4}}()

        frag_end=0
        frag_start = 1

        while frag_start < length(obs_seq) # while we're not at the sequence end
            curr_partition = mask[frag_start,1] #get the partition code of the frag start
            curr_strand = mask[frag_start,2] #get the strand of the frag start

            #JOBID COMPOSED HERE
            frag_start_index = (offsets[o], frag_start, o, curr_partition, curr_strand) #compose an identifying index for this frag

            findnext(!isequal(curr_partition),mask[:,1],frag_start) != nothing ? frag_end = findnext(!isequal(curr_partition),mask[:,1],frag_start) -1 : frag_end = length(obs_seq) #find the next position in the frag that has a different partition mask value from hte current one and set that position-1 to frag end, alternately frag end is end of the overall  sequence 
            frag = obs_seq[frag_start:frag_end] #get the frag bases
            if curr_strand == -1 #if the fragment is reverse stranded
                frag = reverse_complement(frag) #use the reverse complement sequence
            end

            likelihood_jobs[frag_start_index] = frag #put the frag in the jobs index
            frag_start = frag_end + 1 #move on
        end        
    end
    return likelihood_jobs
end

#function for workers to handle BGHMM likelihood calculation queue
function process_BGHMM_likelihood_queue!(lh_jobs::RemoteChannel, output_lhs::RemoteChannel, BGHMM_dict::Dict{String,Tuple{HMM, Int64, Float64}})
    code_partition_dict = BGHMM.get_partition_code_dict(false)

    while isready(lh_jobs)
        jobid, frag = take!(lh_jobs)
        (offset, frag_start, o, partition, strand) = jobid
        partition_BGHMM::MS_HMMBase.HMM = BGHMM_dict[code_partition_dict[partition]][1]
        no_symbols = length(partition_BGHMM.D[1].p)
        order = Int(log(4,no_symbols) - 1)

        order_seq = BGHMM.get_order_n_seqs([frag], order)
        coded_seq = BGHMM.code_seqs(order_seq)
        
        subseq_symbol_lh = get_BGHMM_symbol_lh(coded_seq, partition_BGHMM)
        put!(output_lhs,(jobid,subseq_symbol_lh))
    end
end