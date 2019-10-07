using nnlearn, BGHMM, CLHMM, HMMBase, BioSequences, Distributions, Random, Serialization, Test
import StatsFuns: logsumexp

Random.seed!(1)
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
            @test HMMBase.isprobvec(exp.(source[1][pos,:]))
        end
    end

    permuted_weight_sources=deepcopy(test_sources)
    clean = trues(O,2)
    nnlearn.permute_source_weights!(permuted_weight_sources,100,Uniform(.0001,.02), clean)

    @test permuted_weight_sources != test_sources
    for (s,source) in enumerate(permuted_weight_sources)
        for pos in 1:size(source[1])[1]
            @test HMMBase.isprobvec(exp.(source[1][pos,:]))
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
    obs=Array(transpose(coded_seqs))

    source_pwm = zeros(1,4)
    source_pwm[1,1] = 1
    log_pwm = log.(source_pwm)

    source_stop=5

    @test nnlearn.score_source(obs[1:5,1], log_pwm, source_stop) == [0.0 -Inf; 0.0 -Inf; 0.0 -Inf; 0.0 -Inf; 0.0 -Inf]

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
    obs=Array(transpose(coded_seqs))

    source_pwm = [.7 .1 .1 .1
                  .1 .1 .1 .7
                  .1 .1 .7 .1]
    log_pwm = log.(source_pwm)

    source_start = 1
    source_stop = 10
    
    @test isapprox(exp.(nnlearn.score_source(obs[1:12,1], log_pwm, source_stop)),
        [.7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3])

    #test scoring of multiple obs and sources
    obs=[BioSequences.DNASequence("ATGATGATGATG")
         BioSequences.DNASequence("TGATGATGATGA")]
    order_seqs = BGHMM.get_order_n_seqs(obs, 0)
    coded_seqs = BGHMM.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    
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

    score_mat, source_bitindex, source_wmls = nnlearn.score_sources(sources, obs, [12,12], mix_matrix)

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

    lh_target = -26.821656935021238

    @test nnlearn.weave_scores(o, 12, bg_scores, score_mat, obs_source_indices, obs_source_bitindex, source_wmls, log_motif_expectation, cardinality_penalty) == lh_target

    o1_lh = nnlearn.weave_scores(1, 12, bg_scores, score_mat, findall(mix_matrix[1,:]), source_bitindex[:,mix_matrix[1,:],1], source_wmls, log_motif_expectation, cardinality_penalty)
    o2_lh = nnlearn.weave_scores(2, 12, bg_scores, score_mat, findall(mix_matrix[2,:]), source_bitindex[:,mix_matrix[2,:],2], source_wmls, log_motif_expectation, cardinality_penalty)

    @test nnlearn.IPM_likelihood(sources, [12,12], score_mat, source_bitindex, bg_scores, mix_matrix, source_wmls) == CLHMM.lps(o1_lh,o2_lh)
end

@testset "Full model functions" begin
    source_pwm = [.7 .1 .1 .1
    .1 .1 .1 .7
    .1 .1 .7 .1]

    source_pwm_2 = [.6 .1 .1 .2
    .2 .1 .1 .6
    .1 .2 .6 .1]

    src_length_limits=2:3

    source_priors = nnlearn.assemble_source_priors(3, [source_pwm, source_pwm_2], 4.0, src_length_limits)
    mix_prior=.75

    bg_scores = log.(fill(.5, (12,2)))
    obs=[BioSequences.DNASequence("ATGATGATGATG")
    BioSequences.DNASequence("TGATGATGATGA")]
    order_seqs = BGHMM.get_order_n_seqs(obs, 0)
    coded_seqs = BGHMM.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    obsl=[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]]

    test_model = nnlearn.ICA_PWM_model("test", source_priors, mix_prior, bg_scores, obs, src_length_limits)
    
    sources_target::Vector{Tuple{Matrix{Float64},Int64}} = [([-3.71415 -5.82712 -2.38886 -0.126762; -0.834566 -2.06535 -0.852383 -4.361], 2), ([-0.846903 -1.70222 -1.18194 -2.49741; -1.1486 -2.79613 -1.3878 -0.988191; -6.14759 -3.3357 -0.218022 -1.84412], 1), ([-8.17558 -0.0620875 -3.19203 -3.97233; -3.47489 -3.52956 -1.42089 -0.359224], 1)]
    mix_target = trues(2,3)
    lh_target = -58.01626261977499

    for (s, source) in enumerate(test_model.sources)
        @test isapprox(source[1], sources_target[s][1], atol=9.0e-6)
    end
    @test test_model.mixing_matrix == mix_target
    @test test_model.log_likelihood == lh_target
    
    permuted_model = deepcopy(test_model)
    nnlearn.permute_model!(permuted_model, 1, lh_target, obs, obsl, bg_scores, source_priors, 5)
    @test permuted_model.log_likelihood > test_model.log_likelihood

    reinit_model = deepcopy(test_model)
    nnlearn.reinit_sources!(reinit_model, 1, lh_target, obs, obsl, bg_scores, source_priors, mix_prior, 500)
    @test reinit_model.log_likelihood > test_model.log_likelihood

    uninform_priors = nnlearn.assemble_source_priors(3, Vector{Matrix{Float64}}(), 4.0, src_length_limits)
    ui_model = nnlearn.ICA_PWM_model("ui", uninform_priors, mix_prior, bg_scores, obs, src_length_limits)
    merge_target = ui_model.log_likelihood

    path=randstring()
    test_record = nnlearn.Model_Record(path, test_model.log_likelihood)
    serialize(path,test_model)

    nnlearn.merge_model!([test_record],ui_model,1,merge_target,obs,obsl,bg_scores,500)

    @test ui_model.log_likelihood > merge_target
end

@testset "Ensemble assembly and nested sampling functions" begin
    ensembledir = randstring()

    source_pwm = [.7 .1 .1 .1
    .1 .1 .1 .7
    .1 .1 .7 .1]

    source_pwm_2 = [.6 .1 .1 .2
    .2 .1 .1 .6
    .1 .2 .6 .1]

    src_length_limits=2:3

    source_priors = nnlearn.assemble_source_priors(3, [source_pwm, source_pwm_2], 4.0, src_length_limits)
    mix_prior=.75

    bg_scores = log.(fill(.5, (12,2)))
    obs=[BioSequences.DNASequence("ATGATGATGATG")
    BioSequences.DNASequence("TGATGATGATGA")]
    order_seqs = BGHMM.get_order_n_seqs(obs, 0)
    coded_seqs = BGHMM.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    position_start=1;offsets=[0,0]

    ensemble = nnlearn.Bayes_IPM_ensemble(ensembledir, 10, source_priors, mix_prior, bg_scores, obs, src_length_limits)

    @test length(ensemble.models) == 10
    for model in ensemble.models
        @test -75 < model.log_Li < -40
    end

    models_to_permute = 60

    perm_params = [("permute",(500,9)),("merge",(500)),("init",(500))]
    new_model = nnlearn.run_permutation_routine(ensemble,perm_params,models_to_permute,-40.0)
    @test new_model.log_likelihood > -40.0

    nnlearn.nested_step!(ensemble, perm_params, models_to_permute)
    @test length(ensemble.models) == 10
    @test length(ensemble.log_Li) == 2
    @test ensemble.log_Li[1] < ensemble.log_Li[2]
    @test length(ensemble.log_Xi) == 2
    @test length(ensemble.log_wi) == 2
    @test length(ensemble.log_Liwi) == 2
    @test length(ensemble.log_Zi) == 2
    @test ensemble.log_Zi[1] < ensemble.log_Zi[2]
    @test length(ensemble.Hi) == 2
    @test length(ensemble.retained_posterior_samples) == 1
    @test ensemble.model_counter==12

    final_logZ = nnlearn.nested_sample_posterior_to_convergence!(ensemble, perm_params, models_to_permute)
end