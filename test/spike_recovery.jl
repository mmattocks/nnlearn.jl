#Test of nested sampler accuracy using a HMM-modelled background and synthetic PWM spike-in of two signals, one recurring regularly in few obs, another scattered sparsely across many

using nnlearn, BGHMM, HMMBase, Distributions, Random, Serialization, Distributions

Random.seed!(786)

#CONSTANTS
no_obs=5000
obsl=140

hmm=HMM{Univariate,Float64}([0.4016518533961019, 0.2724399569450827, 0.3138638675018568, 0.012044322156962559], [3.016523036789942e-9 2.860631288328858e-6 0.2299906524188302 0.7700064839333549; 3.0102278323431375e-11 0.7494895424906354 0.23378615437778671 0.016724303101477486; 8.665894321573098e-17 0.2789950381410553 0.7141355461949104 0.006869415664033568; 0.006872526597796038 0.016052425322133648 0.017255179541768192 0.9598198685383041], [Categorical([0.1582723599684065, 0.031949729618356536, 0.653113286526948, 0.15666462388628763]), Categorical([0.4610499748798372, 0.2613013005680122, 0.15161801872560146, 0.12603070582654768]), Categorical([0.08601960130086236, 0.13869192872427524, 0.26945182329686523, 0.5058366466779973]), Categorical([0.3787366527613563, 0.11034604356978756, 0.1119586976058099, 0.3989586060630392])])

struc_sig=[.1 .7 .1 .1
           .1 .1 .1 .7
           .1 .7 .1 .1]
periodicity=10
struc_frac_obs=.35

tata_box=[.05 .05 .05 .85
          .85 .05 .05 .05
          .05 .05 .05 .85
          .85 .05 .05 .05
          .45 .05 .05 .45
          .85 .05 .05 .05
          .45 .05 .05 .45]
tata_frac_obs=.8
tata_recur_range=1:4

combined_ensemble = "/bench/PhD/NGS_binaries/nnlearn/combined_ensemble"

#JOB CONSTANTS
const position_size = 141
const ensemble_size = 500
const no_sources = 2
const source_min_bases = 3
const source_max_bases = 12
@assert source_min_bases < source_max_bases
const source_length_range= source_min_bases:source_max_bases
const mixing_prior = .01
@assert mixing_prior >= 0 && mixing_prior <= 1
const models_to_permute = ensemble_size * 3
const permute_params = [
        ("PSFM",(no_sources, .2, .3)),
        ("PSFM",(no_sources, .8, 1.)),
        ("FM",()),
        ("merge",(no_sources)),
        ("random",(no_sources)),
        ("reinit",(no_sources))
    ]
worker_instruction_rand=true

const prior_wt=10.0

#FUNCTIONS
function setup_obs(hmm, no_obs, obsl)
    obs=vec(Int64.(rand(hmm,obsl)[2]))
    for o in 2:no_obs
        obs=hcat(obs, vec(Int64.(rand(hmm,obsl)[2])))
    end
    obs=vcat(obs,zeros(Int64,1,no_obs))
    return obs
end

function spike_irreg!(obs, source, frac_obs, recur)
    for o in 1:size(obs,2)
        if rand()<frac_obs
            for r in rand(recur)
                rand()<.5 && (source=nnlearn.revcomp_pwm(source))
                pos=rand(1:size(obs,1)-1)
                pwm_ctr=1
                while pos<=size(obs,1)-1&& pwm_ctr<=size(source,1)
                    obs[pos,o]=rand(Categorical(source[pwm_ctr,:]))
                    pos+=1
                    pwm_ctr+=1
                end
            end
        end
    end
end

function spike_struc!(obs, source, frac_obs, periodicity)
    for o in 1:size(obs,2)
        if rand()<frac_obs
            rand()<.5 && (source=nnlearn.revcomp_pwm(source))
            pos=rand(1:periodicity)
            while pos<=size(obs,1)
                pos_ctr=pos
                pwm_ctr=1
                while pos_ctr<=size(obs,1)-1&&pwm_ctr<=size(source,1)
                    obs[pos_ctr,o]=rand(Categorical(source[pwm_ctr,:]))
                    pos_ctr+=1
                    pwm_ctr+=1
                end
                pos+=periodicity
            end
        end
    end
end

function get_BGHMM_lhs(obs,hmm)
    lh_mat=zeros(size(obs,1)-1,size(obs,2))
    for o in 1:size(obs,2)
        obso=zeros(Int64,size(obs,1),1)
        obso[1:end,1] = obs[:,o]
        lh_mat[:,o]=BGHMM.get_BGHMM_symbol_lh(Matrix(transpose(obso)),hmm)
    end
    return lh_mat
end

@info "Setting up synthetic observation set..."
obs=setup_obs(hmm, no_obs, obsl)
spike_irreg!(obs, tata_box, tata_frac_obs, tata_recur_range)
spike_struc!(obs, struc_sig, struc_frac_obs, periodicity)

@info "Calculating background likelihood matrix..."
bg_lhs=get_BGHMM_lhs(obs,hmm)

@info "Assembling source priors..."
source_priors = nnlearn.assemble_source_priors(no_sources, [struc_sig,tata_box], prior_wt, source_length_range)

@info "Assembling ensemble..."
path=randstring()
isfile(string(path,'/',"ens")) ? (ens = deserialize(string(path,'/',"ens"))) :
    (ens = nnlearn.Bayes_IPM_ensemble(path, ensemble_size, source_priors, (falses(0,0), mixing_prior), bg_lhs, obs, source_length_range))

@info "Converging ensemble..."
nnlearn.ns_converge!(ens, permute_params, models_to_permute, .00001, model_display=2, backup=(true,5), wkrand=worker_instruction_rand)

rm(path,recursive=true)