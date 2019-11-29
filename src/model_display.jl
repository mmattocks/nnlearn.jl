#THIS CODE BASED ON N. REDDY'S THICWEED.JL DISPLAY SCRIPT

#CONSTANTS
# each tuple specifies the x, y, fontsize, xscale to print the character
# in a 100x100 box at position 100,100.
charvals_dict = Dict{Char,Tuple}('A'=> (88.5,199.,135.,1.09),
                 'C'=> (79.,196.,129.,1.19),
                 'G'=> (83.,196.,129.,1.14),
                 'T'=> (80.,199.,135.,1.24))

colour_dict = Dict{Char,String}('A'=>"(0,128,0)", #green
                 'C'=>"(0,0,128)", #blue 
                 'G'=>"(255,69,0)", #yellow-brown
                 'T'=>"(150,0,0)")  #red

#char output params for string display of PWMs in nested sampler instrumentation
lwcs_vec=['a','c','g','t']
upcs_vec=['A','C','G','T']
cs_vec=[:green,:blue,:yellow,:red]
thresh_dict=Dict([(0.,'_'),(.25,'.'),(.5,"lwcs"),(1,"upcs")])

source_left_x = 10
xlen = 20
yscale = 250
scale_factor = 0.9
ypixels_per_source = 250
ypad = 10
xpad = 20
xpixels_per_position = 20
fontsize1=60
fontsize2=40

#function to convert ICA_PWM_model sources lists to PWM sequence logo diagrams
function logo_from_model(model::ICA_PWM_model,svg_output::String;freq_sort::Bool=false)
    source_tups = Vector{Tuple{Float64, Int64, Matrix{Float64}}}() #(%of sequences w/ source, prior index, weight matrix)
    mix = model.mix_matrix #o x s
    for (prior, source) in enumerate(model.sources)
        push!(source_tups, (sum(model.mix_matrix[:,prior])/size(model.mix_matrix)[1], prior, exp.(source[1])))
    end

    freq_sort && sort(source_tups)

    file = open(svg_output, "w")
    write(file, svg_header(xpad+xpixels_per_position*maximum([length(source[1]) for source in model.sources]),ypixels_per_source*length(model.sources)+ypad))

    curry = 0
    for (frequency, index, source) in source_tups
        curry += yscale
        font1y = curry-190
        ndig = ndigits(index+1)
        write(file, pwm_to_logo(source,source_left_x,curry,xlen,yscale*scale_factor))
        write(file, "<text x=\"10\" y=\"$font1y\" font-family=\"Helvetica\" font-size=\"$fontsize1\" font-weight=\"bold\" >$(index)</text>")
        write(file, "<text x=\"$(20+fontsize1*0.6*ndig)\" y=\"$(font1y-fontsize1+1.5*fontsize2)\" font-family=\"Helvetica\" font-size=\"$fontsize2\" font-weight=\"bold\" >$(frequency*100) % of sequences</text>\n")   
    end

    write(file, svg_footer())
    @info "Logo written."
    close(file)
end

function svg_header(canvas_size_x, canvas_size_y)
    return """
    <?xml version="1.0" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
    "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <svg width="$canvas_size_x" height="$canvas_size_y"
     version="1.1" xmlns="http://www.w3.org/2000/svg">
    """
end

function pwm_to_logo(source,xpos,ypos,xlen,yscale)
    outstr = ""
    for n in 1:size(source,1)
        outstr *= print_weightvec_at_x_y(source[n,:],xpos+xlen*n,ypos,xlen,yscale)
    end
    outstr *= "\n"
    return outstr
end

function pwmstr_to_io(io,source;log=true)
    log && (source=exp.(source))
    for pos in 1:size(source,1)
        char,color=uni_wvec_params(source[pos,:])
        printstyled(io, char; color=color)
    end
end

function uni_wvec_params(wvec) #assign a single unicode char and color symbol for the most informational position in a position weight vector
    wvec.+=10^-99
    wvec = [x/sum(wvec) for x in wvec]
    infscore=(2.0 + sum([x*log(2,x) for x in wvec]))
    infvec = [x*infscore for x in wvec]
    val,idx=findmax(infvec)
    return char,color=get_char_by_thresh(idx,val)
end
                function get_char_by_thresh(idx,val)
                    char = '?'; seen_thresh=0.
                    for (thresh,threshchar) in thresh_dict
                        val >= thresh && thresh >= seen_thresh && (char=threshchar; seen_thresh=thresh)
                    end
                    char=='?' && println(val)
                    char=="lwcs" && (char=lwcs_vec[idx])
                    char=="upcs" && (char=upcs_vec[idx])
                    color=cs_vec[idx]

                    return char,color
                end





function print_weightvec_at_x_y(wvec, xpos, ypos, xlen, yscale)
    # xlen is the length occupied by that column
    # yscale is the total height for 2 information bits i.e. the
    # maximum height available for a "perfect" nucleotide
    outstr = ""
    basestr = "ACGT"
    wvec.+=10^-99  #prevent log(0) = -Inf
    wvec = [x/sum(wvec) for x in wvec] #renorm
    infscore = (2.0 + sum([x*log(2,x) for x in wvec]))
    if infscore==0.0
        return ""
    end
    wvec = [x*infscore*yscale/2.0 for x in wvec]
    # at this point, the sum of all wvec is a maximum of yscale
    wveclist = [(wvec[n],basestr[n]) for n in 1:4]
    wveclist = sort(wveclist)
    curr_ypos = ypos
    for n in 1:4
        curr_ypos -= wveclist[n][1]
        outstr *= print_char_in_rect(wveclist[n][2],xpos,curr_ypos,xlen,wveclist[n][1])
    end
    return outstr
end

blo = 1

function print_char_in_rect(c,x,y,width,height)
    raw_x, raw_y, raw_fontsize, raw_xscale = charvals_dict[c]
    raw_x = (raw_x*raw_xscale-100)/raw_xscale
    raw_y = raw_y-100
    
    xscale = width/100.0 * raw_xscale
    yscale = height/100.0

    scaled_x = x/xscale + raw_x
    scaled_y = y/yscale + raw_y
    
    return "<text x=\"$scaled_x\" y=\"$scaled_y\" font-size=\"$raw_fontsize\" font-weight=\"bold\" font-family=\"Helvetica\" fill=\"rgb"*colour_dict[c]*"\" transform=\"scale($xscale,$yscale)\">$c</text>\n"
end

function svg_footer()
    return "</svg>"
end
