#function to obtain positional likelihoods for a sequence under a given HMM. Mostly derived from MS_HMMBase.jl mle function
function get_BGHMM_symbol_lh(seq::Matrix{Int64}, hmm::HMM)
    symbol_lhs = zeros(Float64, size(seq))

    lls =  MS_HMMBase.log_likelihoods(hmm, seq) #obtain log likelihoods for sequences and states
    log_α = MS_HMMBase.messages_forwards_log(hmm.π0, hmm.π, lls) #get forward messages
    log_β = MS_HMMBase.messages_backwards_log(hmm.π, lls) # get backwards messages

    #calculate observation probability and γ weights
    T,O,K = size(lls)
    log_γ = fill(-Inf, T,O,K)
    log_pobs = logsumexp(log_α[1,1,:] + log_β[1,1,:])

    for i = 1:K
        for t = 1:findfirst(iszero,lls[:,1,1])-2
            log_γ[t,1,i] = log_α[t,1,i] + log_β[t,1,i] - log_pobs
        end
        t = findfirst(iszero,lls[:,1,1])-1 #correctly assign 0s to logγ t = T - 0 log likelihood at T+1 nulls out ξ numerator and hence γ
        log_γ[t,1,i] = 0
    end

    for pos in 1:length(seq)-1 #iterating down the sequence
        symbol_lh = 0
        for k = 1:K #iterate over states
            state_symbol_lh = log_γ[pos,1,k] + log(hmm.D[k].p[seq[pos]]) #state symbol likelihood is the γ weight * the state symbol probability (log implementation)
            symbol_lh = logsumexp([symbol_lh,state_symbol_lh]) #sum the probabilities over states
        end
        symbol_lhs[pos] = symbol_lh
    end

    return symbol_lhs
end

#function for workers to handle BGHMM likelihood calculation queue
function process_BGHMM_likelihood_queue!(lh_jobs::RemoteChannel, output_lhs::RemoteChannel, BGHMM_dict::Dict{String,Tuple{HMM, Int64, Float64}})
    while isready(lh_jobs)
        partition::String, subseq::Matrix{Int64}, subseq_index::CartesianIndex = take!(lh_jobs)
        partition_BGHMM::MS_HMMBase.HMM = BGHMM_dict[partition][1]
        subseq_symbol_lh::Matrix{Float64} = get_BGHMM_symbol_lh(subseq, partition_BGHMM)
        put!(output_lhs,(subseq_symbol_lh,subseq_index))
    end
end

#function to generate substring BGHMM likelihood jobs from an observation set and mask of BGHMM partitions
function queue_BGHMM_likelihood_calcs(observations::Matrix{Int64}, BGHMM_mask::Matrix{Int64})
    lh_job_queue = RemoteChannel(()->Channel{Tuple{String, Matrix{Int64}, CartesianIndex}}(Inf))
    job_count = 0

    code_partition_dict = BGHMM.get_partition_code_dict(false)

    for o in 1:size(observations)[2] #iterate over observations
        subseqs = Vector{Tuple{String, Matrix{Int64}, CartesianIndex}}() #container for all subsequences in observation
        subseq = Matrix{Int64}(undef,1,1) #container for current subsequence under construction - matrix for MS_HMMBase compatibility
        subseq[1,1] = observations[1,o] #set the first subseq base
        subseq_index = CartesianIndex(1,o) #set the first subseq index as the apropriate observation matrix CartesianInex
        old_partition = BGHMM_mask[1,o] #set for comparison of subsequent base partition codes

        for t in 2:(findfirst(iszero,observations[:,o])-1) #iterate over the length of the observation after the first base
            base_partition = BGHMM_mask[t,o] #get the partition of the base in question
            if base_partition == old_partition
                subseq = vcat(subseq, observations[t,o]) #if the same as last, push the base to the same subseq
            else # if not, terminate the substring with a zero (needed for MS_HMMBase function indexing), push the subseq to the subseqs vector with appropriate information, and begin constructing a new one at this base
                subseq = vcat(subseq, 0)
                push!(subseqs, (code_partition_dict[old_partition], subseq, subseq_index))
                subseq =  Matrix{Int64}(undef,1,1)
                subseq[1,1] = observations[t,o] #set the first base of the new subseq
                subseq_index = CartesianIndex(t,o)
                old_partition = base_partition
            end
        end

        subseq = vcat(subseq, 0) #terminate final/only subseq
        push!(subseqs, (code_partition_dict[old_partition], subseq, subseq_index)) #push final/only subsequence to subseqs vector

        for subseq in subseqs #iterate over the subseqs vector, pushing jobs to the RemoteChannel
            put!(lh_job_queue, subseq)
            job_count += 1
        end
    end

    return lh_job_queue, job_count
end
