@info "Loading test packages..."

using nnlearn, BGHMM, HMMBase, BioSequences, Distributions, Distributed, Random, Serialization, Test
import StatsFuns: logsumexp

@info "Beginning tests..."

Random.seed!(1)
O=1000;S=50

@testset "PWM source prior setup, PWM source initialisation and manipulation functions" begin
    #test dirichlet prior estimation from wm inputs
    wm_input = [.0 .2 .3 .5; .0 .2 .3 .5]
    est_dirichlet_vec = nnlearn.estimate_dirichlet_prior_on_wm(wm_input,4.0)
    @test typeof(est_dirichlet_vec) == Vector{Dirichlet{Float64}}
    for pos in 1:length(est_dirichlet_vec)
        @test isapprox(est_dirichlet_vec[pos].alpha, [0.,.8,1.2,2.0])
    end

    bad_input = wm_input .* 2
    @test_throws DomainError nnlearn.estimate_dirichlet_prior_on_wm(bad_input,4.0)

    wm_input = [.1 .2 .3 .4; .1 .2 .3 .4]
    est_dirichlet_vec = nnlearn.estimate_dirichlet_prior_on_wm(wm_input,4.0)
    @test typeof(est_dirichlet_vec) == Vector{Dirichlet{Float64}}
    for pos in 1:length(est_dirichlet_vec)
        @test est_dirichlet_vec[pos].alpha == [.4,.8,1.2,1.6]
    end

    length_range = 2:2

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
            @test isprobvec(exp.(source[1][pos,:]))
        end
    end

    #test that wm_shift is returning good shifted probvecs
    rando_dist=Dirichlet([.25,.25,.25,.25])
    for i in 1:1000
        wm=rand(rando_dist)
        new_wm=nnlearn.wm_shift(wm,Weibull(1.5,.1))
        @test isprobvec(new_wm)
        @test wm!=new_wm
    end

    #test that legal new sources are generated by permute_source_weights
    permuted_weight_sources=deepcopy(test_sources)
    permuted_weight_sources[1]=nnlearn.permute_source_weights(permuted_weight_sources[1],1.,Weibull(1.5,.1))
    permuted_weight_sources[2]=nnlearn.permute_source_weights(permuted_weight_sources[2],1.,Weibull(1.5,.1))
    @test permuted_weight_sources != test_sources
    for (s,source) in enumerate(permuted_weight_sources)
        for pos in 1:size(source[1],1)
            @test isprobvec(exp.(source[1][pos,:]))
            @test source[1][pos,:] != test_sources[s][1][pos,:]
        end
    end

    #test that get_length_params returns legal length shifts
    lls=1:10
    pr=1:5
    for i in 1:1000
        srcl=rand(1:10)
        sign, permute_length=nnlearn.get_length_params(srcl, lls, pr)
        @test pr[1]<=permute_length<=pr[end]
        @test sign==-1 || sign==1
        @test lls[1]<=(srcl+(sign*permute_length))<=lls[end]
    end

    permuted_length_sources=deepcopy(test_sources)
    permuted_length_sources[1]=nnlearn.permute_source_length(permuted_length_sources[1],test_priors[1],1:3,1:10)
    permuted_length_sources[2]=nnlearn.permute_source_length(permuted_length_sources[2],test_priors[2],1:3,1:10)
    for (s, source) in enumerate(permuted_length_sources)
        @test size(source[1],1) != size(test_sources[1][1],1)
        @test 1<=size(source[1],1)<=3
    end

    info_test_wm=[1. 0. 0. 0.
    .94 .02 .02 .02
    .82 .06 .06 .06
    .7 .1 .1 .1
    .67 .11 .11 .11
    .52 .16 .16 .16
    .4 .2 .2 .2
    .25 .25 .25 .25]

    #test eroding sources by finding most informational position and cutting off when information drops below threshold
    infovec=nnlearn.get_pwm_info(log.(info_test_wm))
    @test infovec==[2.0, 1.5774573308022544, 1.0346297041419121, 0.6432203505529603, 0.5620360019822908, 0.2403724636586433, 0.07807190511263773, 0.0]

    erosion_test_source=(log.([.25 .25 .25 .25
                         .2 .4 .2 .2
                         .7 .1 .1 .1
                         .06 .06 .06 .82
                         .7 .1 .1 .1
                         .25 .25 .25 .25]),1)

    infovec=nnlearn.get_pwm_info(erosion_test_source[1])
    start_idx, end_idx = nnlearn.get_erosion_idxs(infovec, .25, 2:8)
    @test start_idx==3
    @test end_idx==5

    eroded_pwm,eroded_prior_idx=nnlearn.erode_source(erosion_test_source,2:8,.25)
    for pos in 1:size(eroded_pwm,1)
        @test isprobvec(exp.(eroded_pwm[pos,:]))
    end
    @test eroded_prior_idx==3
    @test isapprox(exp.(eroded_pwm),[.7 .1 .1 .1
    .06 .06 .06 .82
    .7 .1 .1 .1])
end

@testset "Mix matrix initialisation and manipulation functions" begin
    #test mix matrix init
    prior_mix_test=nnlearn.init_mix_matrix((trues(2,10),0.0),2, 20)
    @test all(prior_mix_test[:,1:10])
    @test !any(prior_mix_test[:,11:20])

    @test sum(nnlearn.init_mix_matrix((falses(0,0),1.0), O, S)) == O*S
    @test sum(nnlearn.init_mix_matrix((falses(0,0),0.0), O, S)) == 0
    @test 0 < sum(nnlearn.init_mix_matrix((falses(0,0),0.5), O, S)) < O*S

    #test mix matrix decorrelation
    empty_mixvec=falses(O)
    one_mix=nnlearn.mixvec_decorrelate(empty_mixvec,1)
    @test sum(one_mix)==1

    empty_mix = falses(O,S)
    new_mix,clean=nnlearn.mix_matrix_decorrelate(empty_mix, 500)
    @test 0 < sum(new_mix) <= 500
    @test !all(clean)

    full_mix = trues(O,S)
    less_full_mix,clean=nnlearn.mix_matrix_decorrelate(full_mix, 500)

    @test O*S-sum(less_full_mix) <= 500
    @test !all(clean)

    #test matrix similarity and dissimilarity functions
    test_mix=falses(O,S)
    test_mix[1:Int(floor(O/2)),:].=true
    test_idx=3
    compare_mix=deepcopy(test_mix)
    compare_mix[:,test_idx] .= .!compare_mix[:,test_idx]
    @test nnlearn.most_dissimilar(test_mix,compare_mix)==test_idx

    src_mixvec=falses(O)
    src_mixvec[Int(ceil(O/2)):end].=true
    @test nnlearn.most_similar(src_mixvec,compare_mix)==test_idx
end

@testset "Observation setup and model scoring functions" begin
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
    dbl1_srcs=[(log.(source_pwm), 1),(log.(source_pwm), 1)]
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
    penalty_sum = sum(exp.(fill(log_motif_expectation,obs_cardinality)))
    cardinality_penalty=log(1.0-penalty_sum) 

    lh_target = -8.87035766177774

    o1_lh = nnlearn.weave_scores(12, view(bg_scores,:,1), score_mat1, findall(mix_matrix[1,:]), source_wmls, log_motif_expectation, cardinality_penalty)
    @test isapprox(lh_target,o1_lh)
    o2_lh = nnlearn.weave_scores(12, view(bg_scores,:,2), score_mat2, findall(mix_matrix[2,:]), source_wmls, log_motif_expectation, cardinality_penalty)

    lh,cache = nnlearn.IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true,true)
    @test o1_lh==cache[1]
    @test o2_lh==cache[2]
    @test isapprox(nnlearn.lps(o1_lh,o2_lh),lh)
    
    #test source penalization
    dbl_score_mat= nnlearn.score_obs_sources(dbl1_srcs, Vector(obs[:,1]),12,source_wmls)

    naive=nnlearn.weave_scores(12, view(bg_scores,:,1), Vector{Matrix{Float64}}(), Vector{Int64}(), Vector{Int64}(), log_motif_expectation,cardinality_penalty)

    single=nnlearn.weave_scores(12, view(bg_scores,:,1), [dbl_score_mat[1]], [1], [source_wmls[1]], log_motif_expectation, cardinality_penalty)

    double=nnlearn.weave_scores(12, view(bg_scores,:,1), dbl_score_mat, [1,2], source_wmls, log_motif_expectation, cardinality_penalty)

    triple=nnlearn.weave_scores(12, view(bg_scores,:,1), [dbl_score_mat[1] for i in 1:3], [1,2,3], [3 for i in 1:3], log_motif_expectation, cardinality_penalty)

    @test (single-naive) > (double-single) > (triple-double)

    naive_target=-16.635532333438686
    naive = nnlearn.IPM_likelihood(sources, obs, [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]], bg_scores,falses(size(obs)[2],length(sources)))
    @test naive==naive_target

    #test IPM_likelihood clean vector and cache calculations
    baselh,basecache = nnlearn.IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true,true)

    clean=[true,false]

    unchangedlh,unchangedcache=nnlearn.IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true, true, basecache, clean)

    changed_lh,changedcache=nnlearn.IPM_likelihood(sources, obs, [12,12], bg_scores, BitMatrix([true true; false false]), true, true, basecache, clean)

    indep_lh, indepcache=nnlearn.IPM_likelihood(sources,obs,[12,12], bg_scores, BitMatrix([true true; false false]), true, true)

    @test baselh==unchangedlh!=changed_lh==indep_lh
    @test basecache==unchangedcache!=changedcache==indepcache
end

@testset "Model permutation functions" begin
    source_pwm = [.7 .1 .1 .1
    .1 .1 .1 .7
    .1 .1 .7 .1]

    source_pwm_2 = [.6 .1 .1 .2
    .2 .1 .1 .6
    .1 .2 .6 .1]

    pwm_to_erode = [.25 .25 .25 .25
                    .97 .01 .01 .01
                    .01 .01 .01 .97
                    .01 .01 .97 .01
                    .25 .25 .25 .25]

    src_length_limits=2:5

    source_priors = nnlearn.assemble_source_priors(3, [source_pwm, source_pwm_2], 4.0, src_length_limits)
    mix_prior=0.2

    bg_scores = log.(fill(.25, (12,2)))
    obs=[BioSequences.DNASequence("ATGATGATGATG")
    BioSequences.DNASequence("TGATGATGATGA")]
    order_seqs = BGHMM.get_order_n_seqs(obs, 0)
    coded_seqs = BGHMM.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    obsl=[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]]

    test_model = nnlearn.ICA_PWM_model("test", source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)

    duplicate_sources=[(source_pwm,1) for i in 1:3]
    cons_check, cons_idxs = nnlearn.consolidate_check(duplicate_sources)
    @test !cons_check
    @test cons_idxs == [[2,3],[1,3],[1,2]]

    mix=falses(10,3)
    mix[1:2,1].=true
    mix[9:10,2].=true

    con_obs=hcat(obs,obs,obs,obs,obs)
    con_obsl=[12 for i in 1:10]
    con_bg=log.(fill(.5, (12,10)))

    cons_model=nnlearn.consolidate_srcs(cons_idxs, duplicate_sources, mix, con_obs, con_obsl, con_bg, source_priors, test_model.informed_sources, src_length_limits)
    @test cons_model.sources[1]==duplicate_sources[1]
    @test cons_model.sources[2]!=duplicate_sources[2]
    @test cons_model.sources[3]!=duplicate_sources[3]
    @test cons_model.mix_matrix[:,1]==[true, true, false, false, false, false, false, false, true, true]
    @test "consolidate" in cons_model.flags
    @test "FM from candidate" in cons_model.flags
    @test "nofit" in cons_model.flags


    ps_model= nnlearn.permute_source(test_model, test_model.log_Li, obs, obsl, bg_scores, source_priors, 1000,.3,.5)
    @test ps_model.log_Li > test_model.log_Li
    @test ps_model.sources != test_model.sources
    @test ps_model.mix_matrix == test_model.mix_matrix
    @test "PS from test" in ps_model.flags

    pm_model= nnlearn.permute_mix(test_model, test_model.log_Li,obs, obsl, bg_scores, source_priors, 1000)
    @test pm_model.log_Li > test_model.log_Li
    @test pm_model.sources == test_model.sources
    @test pm_model.mix_matrix != test_model.mix_matrix
    @test "PM from test" in pm_model.flags

    badmix_lh=nnlearn.IPM_likelihood(test_model.sources,obs,obsl, bg_scores, trues(2,3))
    badmix=nnlearn.ICA_PWM_model("badmix",test_model.sources, test_model.informed_sources, test_model.source_length_limits, trues(2,3), badmix_lh,Vector{String}())
    psfm_model=nnlearn.perm_src_fit_mix(badmix, badmix.log_Li,obs, obsl, bg_scores, source_priors, 1000)
    @test psfm_model.log_Li > badmix.log_Li
    @test psfm_model.sources != badmix.sources
    @test psfm_model.mix_matrix != badmix.mix_matrix
    @test "PSFM from badmix" in psfm_model.flags

    fm_model=nnlearn.fit_mix(test_model,obs,obsl,bg_scores)
    @test fm_model.log_Li > test_model.log_Li
    @test fm_model.sources == test_model.sources
    @test fm_model.mix_matrix != test_model.mix_matrix
    @test "FM from test" in fm_model.flags
    @test "nofit" in fm_model.flags

    post_fm_psfm=nnlearn.perm_src_fit_mix(fm_model, test_model.log_Li, obs, obsl, bg_scores, source_priors, 1000)
    @test "PSFM from candidate" in post_fm_psfm.flags
    @test "nofit" in post_fm_psfm.flags

    rd_model=nnlearn.random_decorrelate(test_model, test_model.log_Li, obs, obsl, bg_scores, source_priors, 1000)
    @test rd_model.log_Li > test_model.log_Li
    @test rd_model.sources != test_model.sources
    @test rd_model.mix_matrix != test_model.mix_matrix
    @test "RD from test" in rd_model.flags

    rs_model=nnlearn.reinit_src(test_model, test_model.log_Li, obs, obsl, bg_scores, source_priors, (falses(0,0),.5), 1000, false)
    @test rs_model.log_Li > test_model.log_Li
    @test rs_model.sources != test_model.sources
    @test rs_model.mix_matrix != test_model.mix_matrix
    @test "RS from test" in rs_model.flags

    erosion_sources=[(log.(source_pwm),1),(log.(source_pwm_2),1),(log.(pwm_to_erode),1)]

    eroded_mix=trues(2,3)

    erosion_lh=nnlearn.IPM_likelihood(erosion_sources,obs,obsl, bg_scores, eroded_mix)

    erosion_model=nnlearn.ICA_PWM_model("erode", erosion_sources, test_model.informed_sources, test_model.source_length_limits,eroded_mix, erosion_lh, Vector{String}())

    eroded_model=nnlearn.erode_model(erosion_model, erosion_model.log_Li, obs, obsl, bg_scores, source_priors, 1000)
    @test eroded_model.log_Li > erosion_model.log_Li
    @test eroded_model.sources != erosion_model.sources
    @test eroded_model.mix_matrix == erosion_model.mix_matrix
    @test "EM from erode" in eroded_model.flags
    @test eroded_model.sources[1]==erosion_model.sources[1]
    @test eroded_model.sources[2]==erosion_model.sources[2]
    @test eroded_model.sources[3]!=erosion_model.sources[3]
    @test size(eroded_model.sources[3][1],1)==3

    path=randstring()
    test_record = nnlearn.Model_Record(path, rs_model.log_Li)
    serialize(path, rs_model)

    dm_model=nnlearn.distance_merge([test_record], test_model, test_model.log_Li, obs, obsl, bg_scores, source_priors, 1000)
    @test dm_model.log_Li > test_model.log_Li
    @test dm_model.sources != test_model.sources
    @test dm_model.mix_matrix != test_model.mix_matrix
    @test "DM from test" in dm_model.flags

    sm_model=nnlearn.similarity_merge([test_record], test_model, test_model.log_Li, obs, obsl, bg_scores, source_priors, 1000)
    @test sm_model.log_Li > test_model.log_Li
    @test sm_model.sources != test_model.sources
    @test sm_model.mix_matrix != test_model.mix_matrix
    #@test "SM from test" in sm_model.flags

    librarian=addprocs(1)
    @everywhere import nnlearn

    ddm_model=nnlearn.distance_merge(librarian[1], [test_record], test_model, test_model.log_Li, obs, obsl, bg_scores, source_priors, 1000)
    @test ddm_model.log_Li > test_model.log_Li
    @test ddm_model.sources != test_model.sources
    @test ddm_model.mix_matrix != test_model.mix_matrix
    @test "DM from test" in ddm_model.flags

    dsm_model=nnlearn.similarity_merge(librarian[1], [test_record], test_model, test_model.log_Li, obs, obsl, bg_scores, source_priors, 1000)
    @test dsm_model.log_Li > test_model.log_Li
    @test dsm_model.sources != test_model.sources
    @test dsm_model.mix_matrix != test_model.mix_matrix
    #@test "SM from test" in dsm_model.flags
    
    rmprocs(librarian)
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

    source_priors = nnlearn.assemble_source_priors(4, [source_pwm, source_pwm_2], 4.0, src_length_limits)
    mix_prior=.5

    bg_scores = log.(fill(.1, (30,27)))
    obs=[BioSequences.DNASequence("CCGTTGACGATGTGATGAATAATGAAAGAA")
    BioSequences.DNASequence("CCCCGATGATGACCGTTGACCAGATGGATG")
    BioSequences.DNASequence("CCCCGATGATGACCCCGATTTTGAAAAAAA")
    BioSequences.DNASequence("TCATCATGCTGATGATGAATCAGATGAAAG")
    BioSequences.DNASequence("TGATGAATCTGACCCAGATGCCGATTTTGA")
    BioSequences.DNASequence("CCCCGATTTTGATCAGGATGAATAAAGAAA")
    BioSequences.DNASequence("TCATGGGCTGATGAACCGTTGACGATGAAA")
    BioSequences.DNASequence("TCATCCTGCTGACCCCGATTTCAGTGAAAA")
    BioSequences.DNASequence("TGATGAATAAAGTCATCCTGCATGTGAAAA")
    BioSequences.DNASequence("CCGTTGACGATGTGATGAATGATAAAGAAA")
    BioSequences.DNASequence("CCCCGATGATGACCGATGTTGACGATGAAA")
    BioSequences.DNASequence("CCCCGATGATGAATGCCCCGATTTTGAAAA")
    BioSequences.DNASequence("TCATCATGCTGATGATGAATAAAGAAAAAA")
    BioSequences.DNASequence("TGATGAATCTGACCCCGATCAGTTTGAAAA")
    BioSequences.DNASequence("CCCCGATTTTGATCAGATGGATGAATAAAG")
    BioSequences.DNASequence("TCATGGGCTGAACCGTTGACAGCGATGAAA")
    BioSequences.DNASequence("TCATCCTGCTCAGGACCCCGATTTTATGGA")
    BioSequences.DNASequence("TGATGAATCAGAAAGTCATCCTGCATGTGA")
    BioSequences.DNASequence("CCGTTGACCAGGATGTGATGAATAAAGAAA")
    BioSequences.DNASequence("CCCCGATGATGCAGACCGTTGACGATGAAA")
    BioSequences.DNASequence("CCCCGATGACAGTGACCCAGCCGATTTTGA")
    BioSequences.DNASequence("TCATCATGCTGAATGTGATGAATAAAAAAA")
    BioSequences.DNASequence("TGATGATGAATCTGAATGCCCCGATTTTGA")
    BioSequences.DNASequence("CCCCGATATGTTTGATGACAGTGAATAAAG")
    BioSequences.DNASequence("TCATGATGGGCTGAACCGTTGACGATGAAA")
    BioSequences.DNASequence("TCACAGTCCTGCTGACCCCGATTATGTTGA")
    BioSequences.DNASequence("TGATGATGAATAAAGTCATGATCCTGCTGA")
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
        @test -1950 < model.log_Li < -1200
    end

    @test length(sp_ensemble.models) == 200
    for model in sp_ensemble.models
        @test -1950 < model.log_Li < -1200
    end

    assembler=addprocs(1)

    @everywhere using nnlearn

    dist_ensemble=nnlearn.Bayes_IPM_ensemble(assembler, distdir, 150, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    dist_ensemble=nnlearn.Bayes_IPM_ensemble(assembler, distdir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits) #test resumption

    @test length(dist_ensemble.models) == 200
    for model in ensemble.models
        @test -1950 < model.log_Li < -1200
    end

    rmprocs(assembler)
    rm(distdir, recursive=true)

    permute_limit = 900
    param_set = [("source",(10,.25,.5)),("mix",(10)),("merge",(10)),("init",(10))]

    @info "Testing threaded convergence..."
    sp_logZ = nnlearn.ns_converge!(sp_ensemble, param_set, permute_limit, .1, wkrand=true, model_display=4)

    @test length(sp_ensemble.models) == 200
    @test length(sp_ensemble.log_Li) == length(sp_ensemble.log_Xi) == length(sp_ensemble.log_wi) == length(sp_ensemble.log_Liwi) == length(sp_ensemble.log_Zi) == length(sp_ensemble.Hi) == sp_ensemble.model_counter-200
    for i in 1:length(sp_ensemble.log_Li)-1
        @test sp_ensemble.log_Li[i] <= sp_ensemble.log_Li[i+1]
    end
    for i in 1:length(sp_ensemble.log_Zi)-1
        @test sp_ensemble.log_Zi[i] <= sp_ensemble.log_Zi[i+1]
    end
    @test sp_logZ > -1400.0


    @info "Testing multiprocess convergence..."
    @info "Spawning worker pool..."
    librarians=addprocs(1)
    worker_pool=addprocs(2)
    @everywhere using nnlearn,Random
    @everywhere Random.seed!(1)
    
    ####CONVERGE############
    final_logZ = nnlearn.ns_converge!(ensemble, [param_set, param_set], permute_limit, librarians, worker_pool, .1, backup=(true,250), wkrand=[false,true], model_display=4)

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
    @test typeof(final_logZ) == Float64
    @test sp_logZ > -1400.0

    @info "Tests complete!"

    rm(ensembledir, recursive=true)
    rm(spensembledir, recursive=true)
end