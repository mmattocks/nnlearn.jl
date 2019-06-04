#function to obtain positional likelihoods for a sequence under a given HMM
function get_BGHMM_symbol_lh(seq::Vector{Int64}, hmm::HMM)
    symbol_lhs = zeros(Float64, length(seq))

    lls =  MS_HMMBase.log_likelihoods(hmm, seq)
    log_α = MS_HMMBase.messages_forwards_log(hmm.π0, hmm.π, lls)
    log_β = MS_HMMBase.messages_backwards_log(hmm.π, lls)
    T,O,K = size(lls)
    log_γ = fill(-Inf, T,O,K)
    log_pobs = logsumexp(log_α[1,1,:] + log_β[1,1,:])

    for i = 1:K, j = 1:K
        for t = 1:findfirst(iszero,lls[:,1,1])-2
            log_γ[t,1,i] = log_α[t,1,i] + log_β[t,1,i] - log_pobs
        end
        t = findfirst(iszero,lls[:,o,1])-1 #correctly assign 0s to logξ & logγ t = T in each sequence
        log_γ[t,o,i] = 0
    end

    for pos in 1:length(seq)-1
        symbol_lh = 0
        for k = 1:K

        end
    end

end

#function for workers to handle BGHMM likelihood calculation queue
function process_BGHMM_likelihood_queue!(lh_jobs::RemoteChannel, output_lhs::RemoteChannel, BGHMM_dict::Dict{String,Tuple{HMM, Int64, Float64}})
    while isready(lh_jobs)
        partition::String, subseq::Vector{Int64}, subseq_index::CartesianIndex = take!(lh_jobs)
        partition_BGHMM = BGHMM_dict[partition][1]
        subseq_symbol_lh = get_BGHMM_symbol_lh(subseq, partition_BGHMM)
        put!(output_lhs,(subseq_symbol_lh,subseq_index)
    end
end

#function to generate substring BGHMM likelihood jobs from an observation set and mask of BGHMM partitions
function queue_BGHMM_likelihood_calcs(observations::Matrix{Int64}, BGHMM_mask::Matrix{Int64})
    lh_job_queue = RemoteChannel(()->Channel{Tuple{String, Vector{Int64}, CartesianIndex}}(Inf))

    code_partition_dict = BGHMM.get_partition_code_dict(false)

    for o in 1:size(observations)[2] #iterate over observations
        subseqs = Vector{Tuple{String, Vector{Int64}, CartesianIndex}}() #container for all subsequences in observation
        subseq = Vector{Int64}() #container for current subsequence under construction

        push!(subseq, observations[1,o]) #push the first observation base to the subseq
        subseq_index = CartesianIndex(1,o) #set the first subseq index as the apropriate observation matrix CartesianInex
        old_partition = BGHMM_mask[1,o] #set for comparison of subsequent base partition codes

        for t in 2:(findfirst(iszero,observations[:,o])-1) #iterate over the length of the observation after the first base
            base_partition = BGHMM_mask[t,o] #get the partition of the bsae in question
            if base_partition == old_partition
                push!(subseq, observations[t,o]) #if the same as last, push the base to the same subseq
            else # if not, terminate the substring with a zero, push the subseq to the subseqs vector with appropriate information, and begin constructing a new one at this base
                push!(subseq, 0)
                push!(subseqs, (code_partition_dict[old_partition], subseq, subseq_index))
                subseq = Vector{Int64}()
                push!(subseq, observations[t,o])
                subseq_index = CartesianIndex(t,o)
                old_partition = base_partition
            end
        end

        push!(subseq, 0) #terminate final/only subseq
        push!(subseqs, (code_partition_dict[old_partition], subseq, subseq_index)) #push final/only subsequence to subseqs vector

        for subseq in subseqs #iterate over the subseqs vector, pushing jobs to the RemoteChannel
            put!(lh_job_queue, subseq)
        end
    end

    return lh_job_queue
end

#function to provide a log probability score for a sequence given a weight matrix
function seqScorebyWM(logWM::Matrix, seq::Vector{Int})
    #input WM must be in logarithm form
    score::Float64 = 0.0
    @assert size(logWM)[1] == length(seq)
    for position in 1:length(seq)
        score += logWM[position,seq[position]] #access the weight matrix by the numerical position and the integer value of the coded seq at that position
    end
    return score
end
