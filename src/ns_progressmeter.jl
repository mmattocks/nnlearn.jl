#UTILITY PROGRESSMETER, REPORTS WORKER NUMBER AND CURRENT ITERATE
mutable struct ProgressNS{T<:Real} <: AbstractProgress
    interval::T
    dt::Float64
    start_it::Int
    counter::Int
    triggered::Bool
    tfirst::Float64
    tlast::Float64
    printed::Bool        # true if we have issued at least one status update
    desc::AbstractString # prefix to the percentage, e.g.  "Computing..."
    color::Symbol        # default to green
    output::IO           # output stream into which the progress is written
    numprintedvalues::Int   # num values printed below progress in last iteration
    offset::Int             # position offset of progress bar (default is 0)
    information::Float64
    evidence::Float64
    etc::Float64
    contour::Float64
    max_lh::Float64
    naive::Float64
    stepworker::Int64

    function ProgressNS{T}(    naive::Float64,
                               interval::T;
                               dt::Real=0.1,
                               desc::AbstractString="Nested Sampling: ",
                               color::Symbol=:green,
                               output::IO=stderr,
                               offset::Int=0,
                               start_it::Int=1) where T
        tfirst = tlast = time()
        printed = false
        new{T}(interval, dt, start_it, start_it, false, tfirst, tlast, printed, desc, color, output, 0, offset,0.0,0.0,0.0,0.0, 0.0,naive,0)
    end
end

ProgressNS(naive::Float64, interval::Real, dt::Real=0.1, desc::AbstractString="Nested Sampling: ",
         color::Symbol=:green, output::IO=stderr;
         offset::Integer=0, start_it::Integer=1) = ProgressNS{typeof(interval)}(naive, interval, dt=dt, desc=desc, color=color, output=output, offset=offset, start_it=start_it)

ProgressNS(naive::Float64, interval::Real, desc::AbstractString, offset::Integer=0, start_it::Integer=1) = ProgressNS{typeof(interval)}(naive, interval, desc=desc, offset=offset, start_it=start_it)

function update!(p::ProgressNS, contour, max,  val, thresh, info, logz, worker; options...)
    p.contour = contour
    p.max_lh = max
    interval = val - thresh
    step=p.interval-interval
    step_time=(time()-p.tfirst)/(p.counter-p.start_it)
    p.etc= (interval/step)*step_time
    p.interval=interval
    p.information = info
    p.evidence = logz
    p.counter += 1
    p.stepworker = worker
    updateProgress!(p; options...)
end

function updateProgress!(p::ProgressNS; showvalues = Any[], valuecolor = :blue, offset::Integer = p.offset, keep = (offset == 0))
    p.offset = offset
    t = time()
    if p.interval <= 0 && !p.triggered
        p.triggered = true
        if p.printed
            p.triggered = true
            dur = ProgressMeter.durationstring(t-p.tfirst)
            msg = @sprintf "%s Converged. Time: %s (%d iterations). logZ: %s, H: %s" p.desc dur p.counter p.evidence p.information
            print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
            ProgressMeter.move_cursor_up_while_clearing_lines(p.output, p.numprintedvalues)
            ProgressMeter.printover(p.output, msg, p.color)
            ProgressMeter.printvalues!(p, showvalues; color = valuecolor)
            if keep
                println(p.output)
            else
                print(p.output, "\r\u1b[A" ^ (p.offset + p.numprintedvalues))
            end
        end
        return
    end

    if t > p.tlast+p.dt && !p.triggered
        elapsed_time = t - p.tfirst
        msg = @sprintf "%s (Step %i::Wk:%g Contour: %g MELH: %g NLR: %g logZ: %g H: %g CI: %g ETC: %s)" p.desc p.counter p.stepworker p.contour p.max_lh (p.max_lh-p.naive) p.evidence p.information p.interval hmss(p.etc)
        print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
        ProgressMeter.move_cursor_up_while_clearing_lines(p.output, p.numprintedvalues)
        ProgressMeter.printover(p.output, msg, p.color)
        ProgressMeter.printvalues!(p, showvalues; color = valuecolor)
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
                    string(Int(h),":",Int(m),":",Int(ceil(r)))
                end