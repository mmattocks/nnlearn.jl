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

function sample_PWM(prior_dirichlets::Vector{Dirichlet})
    raw_pwm = zeros(4,length(prior_dirichlets))
    for position in 1:length(prior_dirichlets)
        raw_pwm[:,position]=rand(prior_dirichlets[1]).p
    end
    return PWM(raw_pwm)
end

function sample_PWM(uninformative_range::UnitRange{Int64})
    PWM_length = rand(uninformative_range)
    raw_pwm = zeros(4,PWM_length)
    for position in 1:PWM_length
        raw_pwm[:,position]=rand(Dirichlet(ones(4)/4))
    end
    return PWM(raw_pwm)
end

struct ica_pwm_model
    pwm_sources::Vector{PWM}
    mixing_matrix::BitMatrix
end

function init_model_ensemble(no_models::Int64, no_pwm_sources::Int64,mixing_prior::Float64,obs_set_size::Int64; pwm_priors::Vector{Tuple{Vector{Dirichlet},Float64}}=nothing,uninformative_range::UnitRange{Int64})
    ensemble = Vector{ica_pwm_model}()

    for model in 1:no_models
        sources = Vector{PWM}()
        for source in 1:no_pwm_sources

            if pwm_priors != nothing
                prior_roll = rand(1)[1]
                for prior in pwm_priors
                    if prior_roll < prior[2]
                        push!(sources, sample_PWM(prior[1]))
                        break
                    end
                end

            else
                push!(sources, sample_PWM(uninformative_range))
            end
        end

        mix_matrix = BitArray(undef, (obs_set_size, no_pwm_sources))
        for index in eachindex(mix_matrix)
            mix_roll = rand(1)[1]
            if mix_roll < mixing_prior
                mix_matrix[index] = true
            else
                mix_matrix[index] = false
            end
        end

        push!(ensemble, ica_pwm_model(sources, mix_matrix))
    end

    return ensemble
end
