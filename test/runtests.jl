using nnlearn, BGHMM, MS_HMMBase, BioSequences, Distributions, Test
import StatsFuns: logsumexp

O=1000;S=50

@testset "PWM source prior setup, PWM source initialisation and manipulation functions" begin
    #test dirichlet prior estimation from a wm input
    wm_input = [.1 .2 .3 .4
                .1 .2 .3 .4]
    length_range = 2:2
    est_dirichlet_vec = nnlearn.estimate_dirichlet_prior_on_wm(wm_input,4.0)
    @test typeof(est_dirichlet_vec) == Vector{Dirichlet{Float64}}
    for pos in 1:length(est_dirichlet_vec)
        @test est_dirichlet_vec[pos].alpha == [.4,.8,1.2,1.6]
    end

    bad_input = wm_input .* 2
    @test_throws AssertionError nnlearn.estimate_dirichlet_prior_on_wm(bad_input,4.0)

    #test informative/uninformative source prior vector assembly
    test_priors = nnlearn.assemble_source_priors(2, [wm_input], 4.0, length_range)
    @test length(test_priors)  == 2
    for pos in 1:length(test_priors[1])
        @test test_priors[1][pos].alpha == [.4,.8,1.2,1.6]
    end
    for pos in 1:length(test_priors[2])
        @test test_priors[2][pos].alpha == ones(4)/4
    end

    #test source wm initialisation from priors
    test_sources = nnlearn.init_logPWM_sources(test_priors, length_range)
    for source in test_sources
        for pos in 1:size(source[1])[1]
            @test MS_HMMBase.isprobvec(exp.(source[1][pos,:]))
        end
    end

    permuted_weight_sources=deepcopy(test_sources)
    clean = trues(O,2)
    nnlearn.permute_source_weights!(permuted_weight_sources,100,Uniform(.0001,.02), clean)

    @test permuted_weight_sources != test_sources
    for (s,source) in enumerate(permuted_weight_sources)
        for pos in 1:size(source[1])[1]
            @test MS_HMMBase.isprobvec(exp.(source[1][pos,:]))
            @test source[1][pos,:] != test_sources[s][1][pos,:]
        end
    end
    @test sum(clean)==0

    permuted_length_sources=deepcopy(test_sources)
    clean = trues(O,2)
    nnlearn.permute_source_lengths!(permuted_length_sources,test_priors,1,1:3,clean)
    @test permuted_length_sources != test_sources
end

@testset "Mix matrix initialisation and manipulation functions" begin
    #test mix matrix init
    @test sum(nnlearn.init_mixing_matrix(1.0, O, S)) == O*S
    @test sum(nnlearn.init_mixing_matrix(0.0, O, S)) == 0
    @test 0 < sum(nnlearn.init_mixing_matrix(0.5, O, S)) < O*S

    #test mix matrix decorrelation
    empty_mix = falses(O,S)
    clean = trues(O,S)
    nnlearn.mix_matrix_decorrelate!(empty_mix, 500, clean)

    @test sum(empty_mix) == O*S-sum(clean) == 500

    full_mix = trues(O,S)
    clean = trues(O,S)
    nnlearn.mix_matrix_decorrelate!(full_mix, 500, clean)

    @test O*S-sum(full_mix) == O*S-sum(clean) == 500
end

@testset "Observation setup and scoring functions" begin
    #test a trivial scoring example
    obs=[BioSequences.DNASequence("AAAAA")]
    order_seqs = BGHMM.get_order_n_seqs(obs, 0)
    coded_seqs = BGHMM.code_seqs(order_seqs)

    offset=[0]
    source_pwm = zeros(1,4)
    source_pwm[1,1] = 1
    log_pwm = log.(source_pwm)

    source_start=1
    source_stop=5

    @test nnlearn.score_source(coded_seqs[:,1], log_pwm, source_start,source_stop) == [0.0 -Inf; 0.0 -Inf; 0.0 -Inf; 0.0 -Inf; 0.0 -Inf]

    #make sure revcomp_pwm is reversing pwms across both dimensions
    revcomp_test_pwm = zeros(2,4)
    revcomp_test_pwm[1,1] = 1
    revcomp_test_pwm[2,3] = 1
    log_revcomp_test_pwm = log.(revcomp_test_pwm)
    @test nnlearn.revcomp_pwm(log_revcomp_test_pwm) == [-Inf 0 -Inf -Inf
                                                        -Inf -Inf -Inf 0]

    #test a more complicated scoring example
    obs=[BioSequences.DNASequence("ATGATGATGATG")]
    order_seqs = BGHMM.get_order_n_seqs(obs, 0)
    coded_seqs = BGHMM.code_seqs(order_seqs)

    source_pwm = [.7 .1 .1 .1
                  .1 .1 .1 .7
                  .1 .1 .7 .1]
    log_pwm = log.(source_pwm)

    source_start = 1
    source_stop = 10
    
    @test isapprox(exp.(nnlearn.score_source(coded_seqs[:,1], log_pwm, source_start, source_stop)),
        [.7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3])

    #test scoring of multiple obs and sources
    obs=[BioSequences.DNASequence("ATGATGATGATG")
         BioSequences.DNASequence("TGATGATGATGA")]
    order_seqs = BGHMM.get_order_n_seqs(obs, 0)
    coded_seqs = BGHMM.code_seqs(order_seqs)
    
    source_pwm_2 = [.6 .1 .1 .2
                    .2 .1 .1 .6
                    .1 .2 .6 .1]

    target_o1_s1 = [.7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3]
    target_o1_s2 = [.6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2)]
    target_o2_s1 = [.1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3]
    target_o2_s2 = [(.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2)]

    sources = [(log.(source_pwm), 1),(log.(source_pwm_2), 1)]
    position_start = 1
    offsets=[0,0]
    mix_matrix = trues(2,2)

    score_mat, source_bitindex, source_wmls = nnlearn.score_sources(sources, coded_seqs, position_start, offsets, mix_matrix)

    @test isapprox(exp.(score_mat[1,1]),target_o1_s1)
    @test isapprox(exp.(score_mat[1,2]),target_o1_s2)
    @test isapprox(exp.(score_mat[2,1]),target_o2_s1)
    @test isapprox(exp.(score_mat[2,2]),target_o2_s2)

    @test any(source_bitindex[1:2,:,:]) == false
    @test all(source_bitindex[3:12,:,:]) == true

    @test source_wmls == [3,3]

    #test score weaving and IPM likelihood calculations
    o=1
    bg_scores = log.(fill(.5, (12,2)))
    log_motif_expectation = log(0.5 / size(bg_scores)[1])
    obs_source_indices = findall(mix_matrix[o,:])
    obs_source_bitindex = source_bitindex[:,mix_matrix[o,:],o]
    obs_cardinality = length(obs_source_indices)
    cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality))

    @test nnlearn.weave_scores(o, bg_scores, score_mat, obs_source_indices, position_start, obs_source_bitindex, source_wmls, log_motif_expectation, cardinality_penalty) == -26.821656935021238

    o1_lh = nnlearn.weave_scores(1, bg_scores, score_mat, findall(mix_matrix[1,:]), position_start, source_bitindex[:,mix_matrix[1,:],1], source_wmls, log_motif_expectation, cardinality_penalty)
    o2_lh = nnlearn.weave_scores(2, bg_scores, score_mat, findall(mix_matrix[2,:]), position_start, source_bitindex[:,mix_matrix[2,:],2], source_wmls, log_motif_expectation, cardinality_penalty)

    @test nnlearn.IPM_likelihood(sources, score_mat, source_bitindex, bg_scores, mix_matrix, source_wmls) == MS_HMMBase.log_prob_sum(o1_lh,o2_lh)
end

@testset "Full model functions" begin

end

@testset "Ensemble assembly and nested sampling functions" begin

end