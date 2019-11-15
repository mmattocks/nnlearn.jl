using nnlearn, BGHMM, CLHMM, HMMBase, BioSequences, Distributions, Distributed, Random, Serialization, Test
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

    test_mix=nnlearn.init_mixing_matrix((falses(0,0),0.5), 2, 10)

    permuted_weight_sources=deepcopy(test_sources)
    clean = Vector{Bool}(trues(2))
    nnlearn.permute_source_weights!(1,test_mix,permuted_weight_sources,1.,Weibull(1.5,.1), clean)
    nnlearn.permute_source_weights!(2,test_mix,permuted_weight_sources,1.,Weibull(1.5,.1), clean)

    @test permuted_weight_sources != test_sources
    for (s,source) in enumerate(permuted_weight_sources)
        for pos in 1:size(source[1],1)
            @test HMMBase.isprobvec(exp.(source[1][pos,:]))
            if !HMMBase.isprobvec(exp.(source[1][pos,:]))
                println("source $s, pos $pos")
                println(exp.(source[1][pos,:]))
            end
            @test source[1][pos,:] != test_sources[s][1][pos,:]
        end
    end
    @test sum(clean)==0

    permuted_length_sources=deepcopy(test_sources)
    clean = Vector{Bool}(trues(2))
    nnlearn.permute_source_length!(1,test_mix,permuted_length_sources,test_priors,1:3,clean)
    @test size(permuted_length_sources[1][1],1) != size(test_sources[1][1],1)
end

@testset "Mix matrix initialisation and manipulation functions" begin
    #test mix matrix init

    prior_mix_test=nnlearn.init_mixing_matrix((trues(2,10),0.0),2, 20)
    @test all(prior_mix_test[:,1:10])
    @test !any(prior_mix_test[:,11:20])

    @test sum(nnlearn.init_mixing_matrix((falses(0,0),1.0), O, S)) == O*S
    @test sum(nnlearn.init_mixing_matrix((falses(0,0),0.0), O, S)) == 0
    @test 0 < sum(nnlearn.init_mixing_matrix((falses(0,0),0.5), O, S)) < O*S

    #test mix matrix decorrelation
    empty_mix = falses(O,S)
    clean = Vector{Bool}(trues(O))
    nnlearn.mix_matrix_decorrelate!(empty_mix, 500, clean)

    @test sum(empty_mix) > 0

    full_mix = trues(O,S)
    clean = Vector{Bool}(trues(O))
    nnlearn.mix_matrix_decorrelate!(full_mix, 500, clean)

    @test O*S-sum(full_mix) < 500
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
    source_wmls = [size(source[1])[1] for source in sources]

    position_start = 1
    offsets=[0,0]
    mix_matrix = trues(2,2)

    score_mat1 = nnlearn.score_obs_sources(sources, Vector(obs[:,1]), 12, source_wmls)

    @test isapprox(exp.(score_mat1[1]),target_o1_s1)
    @test isapprox(exp.(score_mat1[2]),target_o1_s2)

    score_mat2 = nnlearn.score_obs_sources(sources, Vector(obs[:,2]), 12, source_wmls)

    @test isapprox(exp.(score_mat2[1]),target_o2_s1)
    @test isapprox(exp.(score_mat2[2]),target_o2_s2)

    #test score weaving and IPM likelihood calculations
    o=1
    bg_scores = log.(fill(.5, (12,2)))
    log_motif_expectation = log(0.5 / size(bg_scores)[1])
    obs_source_indices = findall(mix_matrix[o,:])
    obs_cardinality = length(obs_source_indices)
    cardinality_penalty = logsumexp(fill(log_motif_expectation, obs_cardinality))

    lh_target = -26.821656935021238

    o1_lh = nnlearn.weave_scores(12, view(bg_scores,:,1), score_mat1, findall(mix_matrix[1,:]), source_wmls, log_motif_expectation, cardinality_penalty)
    @test isapprox(lh_target,o1_lh)
    o2_lh = nnlearn.weave_scores(12, view(bg_scores,:,2), score_mat2, findall(mix_matrix[2,:]), source_wmls, log_motif_expectation, cardinality_penalty)

    lh,cache = nnlearn.IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true,true)
    @test o1_lh==cache[1]
    @test o2_lh==cache[2]
    @test isapprox(CLHMM.lps(o1_lh,o2_lh),lh)

    naive = nnlearn.IPM_likelihood(sources, obs, [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]], bg_scores,falses(size(obs)[2],length(sources)))

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

    test_model = nnlearn.ICA_PWM_model("test", source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    
    sources_target::Vector{Tuple{Matrix{Float64},Int64}} = [([-3.71415 -5.82712 -2.38886 -0.126762; -0.834566 -2.06535 -0.852383 -4.361], 2), ([-0.846903 -1.70222 -1.18194 -2.49741; -1.1486 -2.79613 -1.3878 -0.988191; -6.14759 -3.3357 -0.218022 -1.84412], 1), ([-8.17558 -0.0620875 -3.19203 -3.97233; -3.47489 -3.52956 -1.42089 -0.359224], 1)]
    mix_target = trues(2,3)
    lh_target = -58.01626261977499

    for (s, source) in enumerate(test_model.sources)
        @test isapprox(source[1], sources_target[s][1], atol=9.0e-6)
    end
    @test test_model.mixing_matrix == mix_target
    @test test_model.log_likelihood == lh_target
    
    ps_model = deepcopy(test_model)
    nnlearn.permute_source!(ps_model, lh_target, obs, obsl, bg_scores, source_priors, 100,1.,.5)
    @test ps_model.log_likelihood > test_model.log_likelihood

    pm_model = deepcopy(test_model)
    nnlearn.permute_mix!(pm_model, lh_target, obs, obsl, bg_scores, source_priors, 100)
    @test (pm_model.log_likelihood > test_model.log_likelihood) || pm_model.log_likelihood===-Inf

    reinit_model = deepcopy(test_model)
    nnlearn.reinit_sources!(reinit_model, lh_target, obs, obsl, bg_scores, source_priors, (falses(0,0),mix_prior), 500)
    @test reinit_model.log_likelihood > test_model.log_likelihood

    uninform_priors = nnlearn.assemble_source_priors(3, Vector{Matrix{Float64}}(), 4.0, src_length_limits)
    ui_model = nnlearn.ICA_PWM_model("ui", uninform_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    merge_target = ui_model.log_likelihood

    path=randstring()
    test_record = nnlearn.Model_Record(path, test_model.log_likelihood)
    serialize(path, test_model)

    nnlearn.merge_model!(1,[test_record],ui_model,merge_target,obs,obsl,bg_scores,500)

    @test ui_model.log_likelihood > merge_target

    rm(path)
end

@testset "Ensemble assembly and nested sampling functions" begin
    ensembledir = randstring()
    spensembledir = randstring()
    distdir = randstring()

    source_pwm = [.7 .1 .1 .1
    .1 .1 .1 .7
    .1 .1 .7 .1]

    source_pwm_2 = [.6 .1 .1 .2
    .2 .1 .1 .6
    .1 .2 .6 .1]

    src_length_limits=2:12

    source_priors = nnlearn.assemble_source_priors(60, [source_pwm, source_pwm_2], 4.0, src_length_limits)
    mix_prior=.5

    bg_scores = log.(fill(.5, (12,9)))
    obs=[BioSequences.DNASequence("CCGTTGACGATG")
    BioSequences.DNASequence("CCCCGATGATGA")
    BioSequences.DNASequence("CCCCGATGATGA")
    BioSequences.DNASequence("TCATCATGCTGA")
    BioSequences.DNASequence("TGATGAATCTGA")
    BioSequences.DNASequence("CCCCGATTTTGA")
    BioSequences.DNASequence("TCATGGGCTGAA")
    BioSequences.DNASequence("TCATCCTGCTGA")
    BioSequences.DNASequence("TGATGAATAAAG")
    ]
    
    order_seqs = BGHMM.get_order_n_seqs(obs, 0)
    coded_seqs = BGHMM.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    position_start=1;offsets=[0,0]

    ensemble = nnlearn.Bayes_IPM_ensemble(ensembledir, 150, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    ensemble = nnlearn.Bayes_IPM_ensemble(ensembledir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits) #test resumption

    sp_ensemble = nnlearn.Bayes_IPM_ensemble(spensembledir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)

    @test length(ensemble.models) == 200
    for model in ensemble.models
        @test -150 < model.log_Li < -25
    end

    @test length(sp_ensemble.models) == 200
    for model in sp_ensemble.models
        @test -150 < model.log_Li < -25
    end

    assembler=addprocs(1)

    @everywhere using nnlearn

    dist_ensemble=nnlearn.Bayes_IPM_ensemble(assembler, distdir, 150, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    dist_ensemble=nnlearn.Bayes_IPM_ensemble(assembler, distdir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits) #test resumption

    @test length(dist_ensemble.models) == 200
    for model in ensemble.models
        @test -150 < model.log_Li < -25
    end

    rmprocs(assembler)
    rm(distdir, recursive=true)

    permute_limit = 900
    param_set = [("source",(10,.25,.5)),("mix",(10)),("merge",(10)),("init",(10))]

    @info "Testing threaded convergence..."
    sp_logZ = nnlearn.ns_converge!(sp_ensemble, param_set, permute_limit, .1)

    @test length(sp_ensemble.models) == 200
    @test length(sp_ensemble.log_Li) == length(sp_ensemble.log_Xi) == length(sp_ensemble.log_wi) == length(sp_ensemble.log_Liwi) == length(sp_ensemble.log_Zi) == length(sp_ensemble.Hi) == sp_ensemble.model_counter-200
    for i in 1:length(sp_ensemble.log_Li)-1
        @test sp_ensemble.log_Li[i] <= sp_ensemble.log_Li[i+1]
    end
    for i in 1:length(sp_ensemble.log_Zi)-1
        @test sp_ensemble.log_Zi[i] <= sp_ensemble.log_Zi[i+1]
    end
    @test sp_logZ > -104.0


    @info "Testing multiprocess convergence (error expected)..."
    @info "Spawning worker pool..."
    librarians=addprocs(1)
    worker_pool=addprocs(1)
    @everywhere using nnlearn,Random
    @everywhere Random.seed!(1)
    
    ####CONVERGE############
    final_logZ = nnlearn.ns_converge!(ensemble, [param_set], permute_limit, librarians, worker_pool, 25., backup=(true,250))

    rmprocs(worker_pool)
    rmprocs(librarians)

    @test length(ensemble.models) == 200
    @test length(ensemble.log_Li) == length(ensemble.log_Xi) == length(ensemble.log_wi) == length(ensemble.log_Liwi) == length(ensemble.log_Zi) == length(ensemble.Hi) == ensemble.model_counter-200
    for i in 1:length(ensemble.log_Li)-1
        @test ensemble.log_Li[i] <= ensemble.log_Li[i+1]
    end
    for i in 1:length(ensemble.log_Zi)-1
        @test ensemble.log_Zi[i] <= ensemble.log_Zi[i+1]
    end
    @test typeof(final_logZ) == nnlearn.Bayes_IPM_ensemble

    @info "Worker exhaustion confirmed. Tests done!"

    rm(ensembledir, recursive=true)
    rm(spensembledir, recursive=true)
end