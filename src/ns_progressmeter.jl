#UTILITY PROGRESSMETER, REPORTS WORKER NUMBER AND CURRENT ITERATE
mutable struct ProgressNS{T<:Real} <: AbstractProgress
    interval::T
    dt::Float64
    start_it::Int
    counter::Int
    triggered::Bool
    tfirst::Float64
    tlast::Float64
    tstp::Float64
    printed::Bool        # true if we have issued at least one status update
    desc::AbstractString # prefix to the percentage, e.g.  "Computing..."
    color::Symbol        # default to green
    output::IO           # output stream into which the progress is written
    numprintedvalues::Int   # num values printed below progress in last iteration
    offset::Int             # position offset of progress bar (default is 0)
    total_step::Float64
    information::Float64
    etc::Float64
    contour::Float64
    max_lh::Float64
    naive::Float64
    li_dist::Vector{Float64}
    stepworker::Int64
    workers::Vector{Int64}
    wk_totals::Vector{Int64}
    total_li_delta::Vector{Float64}
    wk_li_delta::Matrix{Float64}
    wk_instruction::Vector{Int64}
    model_exhaust::Vector{Int64}
    wk_eff::Vector{Vector{Float64}}
    wk_jobs::Vector{String}
    SMiMeI::Vector{Int64}
    mean_stp_time::Float64
    eff_iterates::Int64
    no_displayed_srcs::Int64
    sources::Vector{Tuple{Matrix{Float64},Int64}}
    mix::BitMatrix  

    function ProgressNS{T}(    naive::Float64,
                               interval::T,
                               workers::Vector{Int64};
                               dt::Real=0.1,
                               eff_iterates::Int64,
                               desc::AbstractString="Nested Sampling::",
                               color::Symbol=:green,
                               output::IO=stderr,
                               offset::Int=0,
                               start_it::Int=1,
                               no_displayed_srcs::Int64=0) where T
        tfirst = tlast = time()
        printed = false
        new{T}(interval,
         dt,
         start_it,
         start_it,
         false,
         tfirst,
         tlast,
         0., 
         printed, 
         desc,
         color,
         output,
         0,
         offset,
         0.,0.,0.,0.,0.,
         naive,
         [0.],
         0,
         workers,
         zeros(Int64,length(workers)),
         zeros(length(workers)),
         zeros(length(workers), 65),
         zeros(Int64,length(workers)),
         zeros(Int64,length(workers)),
         [[0.] for i in 1:length(workers)],
         ["none" for worker in 1:length(workers)],
         zeros(Int64,4),
         0.,
         eff_iterates,
         no_displayed_srcs,
         [(zeros(0,0),0)],
         falses(0,0))
    end
end

ProgressNS(naive::Float64, interval::Real, workers::Vector{Int64}, dt::Real=0.1, desc::AbstractString="Nested Sampling::",
         color::Symbol=:green, output::IO=stderr, offset::Integer=0, start_it::Integer=1, eff_iterates=250) = 
            ProgressNS{typeof(interval)}(naive, interval, workers, dt=dt, eff_iterates=eff_iterates, desc=desc, color=color, output=output, offset=offset, start_it=start_it)

ProgressNS(naive::Float64, interval::Real, workers::Vector{Int64}, desc::AbstractString, offset::Integer=0, start_it::Integer=1; eff_iterates=250, no_displayed_srcs=1) = ProgressNS{typeof(interval)}(naive, interval, workers, desc=desc, offset=offset, start_it=start_it, eff_iterates=eff_iterates, no_displayed_srcs=no_displayed_srcs)

function update!(p::ProgressNS, contour, max, val, thresh, info, li_dist, worker, wk_time, job, model, old_li, new_li, instruction; sources=[(zeros(0,0),0)], bitmatrix=falses(0,0), options...)
    
    instruction == "source" && (p.SMiMeI[1]+=1)
    instruction == "mix" && (p.SMiMeI[2]+=1)
    instruction == "merge" && (p.SMiMeI[3]+=1)
    instruction == "init" && (p.SMiMeI[4]+=1)

    p.counter += 1
    stps_elapsed=p.counter-p.start_it
    p.tstp=time()-p.tlast
    p.mean_stp_time=(p.tlast-p.tfirst)/stps_elapsed

    widx=findfirst(isequal(worker), p.workers)
    p.wk_totals[widx]+=1
    li_delta=new_li-old_li
    p.total_li_delta[widx]+=new_li-old_li

    p.wk_li_delta[widx,1:end-1]=p.wk_li_delta[widx,2:end]
    p.wk_li_delta[widx,end]=li_delta

    push!(p.wk_eff[widx],li_delta/wk_time)
    length(p.wk_eff[widx])>p.eff_iterates && (p.wk_eff[widx]=p.wk_eff[widx][end-p.eff_iterates+1:end]) #keep worker efficiency array under iterate size limit
    p.wk_instruction[widx]=job; p.model_exhaust[widx]=model
    p.wk_jobs[widx]=instruction

    p.contour = contour
    p.max_lh = max

    interval = val - thresh
    step = p.interval - interval
    !isinf(step) && step>0 && (p.total_step+=step)

    p.etc= (p.interval/(p.total_step/stps_elapsed))*p.mean_stp_time

    p.interval=interval
    p.information = info
    p.li_dist=li_dist
    p.stepworker = worker
    p.sources=sources
    p.mix=bitmatrix
    updateProgress!(p; options...)
end

function updateProgress!(p::ProgressNS; showvalues = Any[], valuecolor = :blue, offset::Integer = p.offset, keep = (offset == 0))
    p.offset = offset
    t = time()
    if p.interval <= 0 && !p.triggered
        p.triggered = true
        if p.printed
            p.triggered = true
            wk_msgs = [@sprintf "Wk:%g:: I:%i, %s M:%i S:%.2f" p.workers[widx] p.wk_instruction[widx] p.wk_jobs[widx] p.model_exhaust[widx] (p.wk_totals[widx]/(p.counter-p.start_it)) for widx in 1:length(p.workers)]
            wk_inst=UnicodePlots.boxplot(wk_msgs, p.wk_eff, title="Worker Diagnostics", xlabel="Likelihood surface velocity (sec^-1)", color=:magenta)
    
            lh_heatmap=UnicodePlots.heatmap(p.wk_li_delta[end:-1:1,:], xoffset=-size(p.wk_li_delta,2)-1, colormap=:viridis, title="Worker lhΔ history", xlabel="Lh stride/step")

            dur = ProgressMeter.durationstring(t-p.tfirst)
            msg1 = @sprintf "%s Converged. Time: %s (%d iterations). H: %s" p.desc dur p.counter p.information
            print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
            msg2 = @sprintf "Ensemble Stats:: Contour: %g MaxLH: %g Max/Naive: %g H: %g" p.contour p.max_lh (p.max_lh-p.naive) p.information
            hist=UnicodePlots.histogram(p.li_dist, title="Ensemble Likelihood Distribution", color=:green)
            #p.numprintedvalues=nrows(hist.graphics)
            srclines=p.no_displayed_srcs+1
            p.numprintedvalues=nrows(wk_inst.graphics)+nrows(lh_heatmap.graphics)+nrows(hist.graphics)+16+srclines
            print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
            ProgressMeter.move_cursor_up_while_clearing_lines(p.output, p.numprintedvalues)
            show(p.output, wk_inst)
            print(p.output, "\n")
            show(p.output, lh_heatmap)
            print(p.output, "\n")
            p.no_displayed_srcs>0 && printsources(p)
            ProgressMeter.printover(p.output, msg1, :magenta)
            print(p.output, "\n")
            ProgressMeter.printover(p.output, msg2, p.color)
            print(p.output, "\n")
            show(p.output, hist)
            print(p.output, "\n")

            if keep
                println(p.output)
            else
                print(p.output, "\r\u1b[A" ^ (p.offset + p.numprintedvalues))
            end
        end
        return
    end

    if t > p.tlast+p.dt && !p.triggered
        wk_msgs = [@sprintf "Wk:%g:: I:%i, %s M:%i S:%.2f" p.workers[widx] p.wk_instruction[widx] p.wk_jobs[widx] p.model_exhaust[widx] (p.wk_totals[widx]/(p.counter-p.start_it)) for widx in 1:length(p.workers)]
        wk_inst=UnicodePlots.boxplot(wk_msgs, p.wk_eff, title="Worker Diagnostics", xlabel="Likelihood surface velocity (sec^-1)", color=:magenta)

        lh_heatmap=UnicodePlots.heatmap(p.wk_li_delta[end:-1:1,:], xoffset=-size(p.wk_li_delta,2)-1, colormap=:viridis, title="Worker lhΔ history", xlabel="Lh stride/step")
        
        msg1 = @sprintf "%s Step %i::Wk:%g: S|Mi|Me|I:%s|%s|%s|%s T μ,Δ: %s,%s CI: %g ETC: %s" p.desc p.counter p.stepworker p.SMiMeI[1] p.SMiMeI[2] p.SMiMeI[3] p.SMiMeI[4] hmss(p.mean_stp_time) hmss(p.tstp-p.mean_stp_time) p.interval hmss(p.etc)
        msg2 = @sprintf "Ensemble Stats:: Contour: %g MaxLH: %g Max/Naive: %g H: %g" p.contour p.max_lh (p.max_lh-p.naive) p.information

        hist=UnicodePlots.histogram(p.li_dist, title="Ensemble Likelihood Distribution", color=:green)

        #p.numprintedvalues=nrows(wk_inst.graphics)+nrows(hist.graphics)+nrows(lh_heatmap.graphics)+1
        srclines=p.no_displayed_srcs+1
        p.numprintedvalues=nrows(wk_inst.graphics)+nrows(lh_heatmap.graphics)+nrows(hist.graphics)+16+srclines
        print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
        ProgressMeter.move_cursor_up_while_clearing_lines(p.output, p.numprintedvalues)
        show(p.output, wk_inst)
        print(p.output, "\n")
        show(p.output, lh_heatmap)
        print(p.output, "\n")
        p.no_displayed_srcs>0 && printsources(p)
        ProgressMeter.printover(p.output, msg1, :magenta)
        print(p.output, "\n")
        ProgressMeter.printover(p.output, msg2, p.color)
        print(p.output, "\n")
        show(p.output, hist)
        print(p.output, "\n")

        print(p.output, "\r\u1b[A" ^ (p.offset + p.numprintedvalues))

        # Compensate for any overhead of printing. This can be
        # especially important if you're running over a slow network
        # connection.
        p.tlast = t + 2*(time()-t)
        p.printed = true
    end
end

                function hmss(dt)
                    isnan(dt) && return "NaN"
                    (h,r) = divrem(dt,60*60)
                    (m,r) = divrem(r, 60)
                    (isnan(h)||isnan(m)||isnan(r)) && return "NaN"
                    string(Int(h),":",Int(m),":",Int(ceil(r)))
                end

                function printsources(p; freqsort=true)
                    printidxs=Vector{Int64}()
                    printsrcs=Vector{Matrix{Float64}}()
                    printfreqs=Vector{Float64}()

                    if freqsort
                        freqs=vec(sum(p.mix,dims=1)); total=size(p.mix,1)
                        sortfreqs=sort(freqs,rev=true)
                        sortidxs=sortperm(freqs)
                        for srcidx in 1:p.no_displayed_srcs
                            push!(printidxs, sortidxs[srcidx])
                            push!(printsrcs, p.sources[sortidxs[srcidx]][1])
                            push!(printfreqs, sortfreqs[srcidx]/total)
                        end
                    else
                        freqs=vec(sum(p.mix,dims=1)); total=size(p.mix,1)
                        for srcidx in 1:p.no_displayed_srcs
                            push!(printidxs, srcidx)
                            push!(printsrcs, p.sources[srcidx][1])
                            push!(printfreqs, freqs[srcidx]/total)
                        end
                    end

                    printstyled(p.output, "MLE Top Sources\n", bold=true)

                    for src in 1:p.no_displayed_srcs
                        print(p.output, "S$(printidxs[src]), $(printfreqs[src])%: ")
                        pwmstr_to_io(p.output, printsrcs[src])
                        print(p.output, "\n")
                    end
                end
