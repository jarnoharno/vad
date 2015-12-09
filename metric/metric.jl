module Metric

# for @match
using Match
# for isexpr
using Base.Meta

@enum(Score,
# Score values
TN, # true negative
FP, # false positive
FN, # false negative
TP, # true positive

# Score subvalues for FP
I,  # insertion
M,  # merge
OS, # overfill start
OE, # overfill end

# Score subvalues for FN
D,  # deletion
F,  # fragmenting
US, # underfill start
UE, # underfill end

# Invalid value, represents values outside the trace (beginning and end)
NA)

# read event sequence as a list of timestamps
function read_seq(path)
    readdlm(path,' ',Float64)[:]
end

function compute_scores(actual, predicted)
    #extended_scores(initial_scores(read_seq(actual),read_seq(predicted)))
    initial_scores(read_seq(actual),read_seq(predicted))
end

function initial_scores(actual::AbstractString, predicted::AbstractString)
    initial_scores(read_seq(actual),read_seq(predicted))
end

function initial_scores(actual::Vector{Float64}, predicted::Vector{Float64})
    # last timestamp
    t_end = max(actual[end],predicted[end])
    # merge
    seq = vcat(
        [(t,1) for t in actual[1:end-1]],
        [(t,2) for t in predicted[1:end-1]])
    # append final timestamp
    push!(seq,(t_end,1))
    sort!(seq)
    # score value index
    score_index = [TN FP; FN TP]
    score_state = [1,1]
    t = 0.0
    segs = Array(Tuple{Score,Float64},0)
    for s in seq
        dt = s[1]-t
        score = score_index[score_state...]
        # update
        t = s[1]
        score_state[s[2]] = score_state[s[2]] == 1 ? 2 : 1
        # ignore empty segments
        if dt > 0.0
            # eliminate duplicates
            if !isempty(segs) && segs[end][1] == score
                segs[end] = (score,segs[end][2]+dt)
            else
                push!(segs,(score,dt))
            end
        end
    end
    segs
end

# convert variables in an expression into constant literals
function literalize(m)
    if isa(m,Symbol)
        return m == :_ ? m : eval(m)
    end
    if isexpr(m,:line)
        return m
    end
    for i=(m.head == :call ? 2 : 1):length(m.args)
        m.args[i] = literalize(m.args[i])
    end
    return m
end

# version of match macro that converts variables into constant literals
macro cmatch(x,v)
    w = literalize(v)
    :(@match $x $w)
end

# map consecutive score values to segment values (from Ward, 2011, Fig. 3)
segment_score(x) = @cmatch x begin
    (NA,FP,TN||FN)=>I; (TP,FP,TP)=>M; (TN||FN,FP,TN||FN)=>I; (TN||FN,FP,NA)=>I
    (NA,FN,TN||FN)=>D; (TP,FN,TP)=>F; (TN||FP,FN,TN||FP)=>D; (TN||FP,FN,NA)=>D
    (NA,FP,TP)=>OS; (TN||FN,FP,TP)=>OS; (TP,FP,TN||FN)=>OE; (TP,FP,NA)=>OE
    (NA,FN,TP)=>US; (TN||FP,FN,TP)=>US; (TP,FN,TN||FP)=>UE; (TP,FN,NA)=>UE
    (_,TN,_)=>TN; (_,TP,_)=>TP; _=>NA
end

# convert score values to segment values
function extended_scores(x)
    z = vcat((NA,1.0),x...,(NA,1.0))
    [(segment_score((z[i][1],z[i+1][1],z[i+2][1])),x[i][2]) for i=1:length(x)]
end

# compute sums of segment values
function summarize(scores)
    d = Dict{Metric.Score,Float64}([s=>0.0 for s in instances(Metric.Score)])
    # remove zero scores
    pop!(d,NA)
    #pop!(d,FP)
    #pop!(d,FN)
    for s in scores
        d[s[1]] += s[2]
    end
    d
end

function run(id, actual, predicted)
    scores = compute_scores(actual,predicted)
    summary = summarize(scores)
    for (k,v) in summary
        @printf("%s %s %f\n",id,k,v)
    end
end

end

if !isinteractive()
    Metric.run(ARGS...)
end
