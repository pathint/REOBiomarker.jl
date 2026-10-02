using Random
using Statistics

# ==============================================================================
# Majority-voting strategy, BQC-score grouped selection 
# ==============================================================================

"""
    _fit_voting_strategy(X, bqc_scores, labels, gene_ids, pairs_idx, cfg)

Train pipeline for the voting strategy. Return a trained REOModel.

Selection protocol (controlled by `cfg.mode`:
 * "soft" mode:
  1. `bqc_scores` is sorted from high to low and takes discrete values;
     keep whole score groups until the total number of pairs reaches `cfg.target_n`;
  2. if the number of selected pairs is even, randomly drop one pair from the
     lowest-score group;
  * "hard" mode:
  1. gene pairs are chosen from high to low `bqc_scores` until the total number of 
     pairs reaches `cfg.target_n`;
  2. if the number of the last `bqc_scores` group will exceed `cfg.target_n`, 
     gene pairs are randomly selected from the grup.
 * "auc" mode:
  1. candidate gene pairs are collected exactly as in "soft" mode (whole score
     groups until the total number >= `cfg.target_n`);
  2. if the number of candidates exceeds `cfg.target_n`, candidates are ranked by
     their univariate AUC benefit and the top `cfg.target_n` pairs are kept;

  3. report train-set accuracy / AUC / MCC of the majority-vote classifier;
  4. no further feature optimisation is performed.
"""
function _fit_voting_strategy(X, bqc_scores, labels, gene_ids, pairs_idx, cfg)
    y = UInt8.(labels)

    cfg.verbose && 
	println(">>> Select gene pairs by grouped BQC score (mode = $(cfg.mode)) ...")

    n_all = length(bqc_scores)
    n_all == 0 && error("bqc_scores is empty: no gene pair available.")

    n_idx = length(pairs_idx)
    n_x   = size(X, 2)
    n_x == n_all == n_idx || error("bqc_scores does not match with pairs_idx")

    # (1) 按 cfg.mode 选取基因对
    rng = MersenneTwister(cfg.seed)

    if cfg.mode == "soft"
        # soft: 整组选取直到总数 >= cfg.target_n
        selected = Int[]
        i = 1
        while i <= n_all
            s = bqc_scores[i]
            j = i
            while j <= n_all && bqc_scores[j] == s
                j += 1
            end
            append!(selected, i:(j - 1))          # 同一 score 值为一组，整组入选
            i = j
            length(selected) >= cfg.target_n && break
        end

    elseif cfg.mode == "auc"
        # auc: 候选生成与 soft 相同（整组选取直到总数 >= cfg.target_n）
        groups     = Vector{Vector{Int}}()
        candidates = Int[]
        i = 1
        while i <= n_all
            s = bqc_scores[i]
            j = i
            while j <= n_all && bqc_scores[j] == s
                j += 1
            end
            g = collect(i:(j - 1))
            push!(groups, g)
            append!(candidates, g)
            i = j
            length(candidates) >= cfg.target_n && break
        end

        if length(candidates) > cfg.target_n
            # 超出 target_n: 是按“加入后的分类 AUC”从最后一组中选
            base = Int[]                          # 更高分值基因对全部选中
            for g in groups[1:(end - 1)]
                append!(base, g)
            end
            lastgrp = groups[end]                 # 最后一个同 bqc_score 的组
            need    = cfg.target_n - length(base) # 还需从 lastgrp 中选的数目

            # 对 lastgrp 中每个基因对: 计算 (base ∪ {c}) 的多数投票分类 AUC
            base_votes = isempty(base) ? zeros(Int16, size(X, 1)) :
                         Int16.(vec(sum(Int.(X[:, base]), dims = 2)))
            cand_auc   = Vector{Float64}(undef, length(lastgrp))
            for (t, c) in enumerate(lastgrp)
                v = base_votes .+ Int16.(X[:, c])
                cand_auc[t] = calc_auc(v, y)      # 单个基因对加入后的分类 AUC
            end

            # 选AUC最高的 need 个；AUC一致时随机选择
            tiebreak = rand(rng, length(lastgrp))
            order    = sort(1:length(lastgrp); by = t -> (-cand_auc[t], tiebreak[t]))
            picked   = lastgrp[order[1:need]]
            selected = [base; picked]
            cfg.verbose && println(">>> auc mode: $(length(candidates)) candidates, " *
                                   "keep $(length(base)) higher-score pairs and " *
                                   "$need of $(length(lastgrp)) by AUC")
        else
            selected = candidates
        end

    elseif cfg.mode == "hard"
        # hard: 从最高分值组逐组选取，直到数目 == cfg.target_n
        selected = Int[]
        i = 1
        while i <= n_all && length(selected) < cfg.target_n
            s = bqc_scores[i]
            j = i
            while j <= n_all && bqc_scores[j] == s
                j += 1
            end
            group = collect(i:(j - 1))            # 当前同一 score 值的一组
            need  = cfg.target_n - length(selected)

            if length(group) <= need
                append!(selected, group)          # 整组入选还不够，继续下一组
            else
                # 该组加入后将超过 cfg.target_n：在该组中随机选取 need 个
                picked = shuffle(rng, copy(group))[1:need]
                append!(selected, picked)
            end
            i = j
        end

    else
        error("Unsupported mode: $(cfg.mode), choose \"soft\", \"hard\" or \"auc\"")
    end

    # (2) soft 模式: 总数为偶数时，随机删除最低分值组中的一个
    if cfg.mode == "soft" && iseven(length(selected))
        s_min  = bqc_scores[selected[end]]
        lowest = filter(idx -> bqc_scores[idx] == s_min, selected)
        drop   = rand(rng, lowest)
        selected = filter(x -> x != drop, selected)
        cfg.verbose && println(">>> Even-sized subset: randomly dropped pair index " *
                               " $drop (BQC = $s_min)")
    end

    n_top = length(selected)
    cfg.verbose && println(">>> $n_top gene pairs were selected.")

    final_named_pairs = Vector{Tuple{String, String}}()
    for idx in selected
        g1, g2 = pairs_idx[idx]
        push!(final_named_pairs, (gene_ids[g1], gene_ids[g2]))
    end

    # 等权赋值: 每个基因对的权重相同
    voting_weights = fill(1.0 / n_top, n_top)

    # (3) 训练集上的多数投票分类指标
    votes = Int16.(vec(sum(Int.(X[:, selected]), dims = 2)))
    tau   = cld(n_top, 2)                                  # 多数投票阈值

    acc = calc_accuracy(votes, y, tau)
    auc = calc_auc(votes, y)
    mcc = calc_mcc_at(votes, y, tau)

    #cfg.verbose && println(">>> Best threshold for majority vote: >= $tau votes")
    cfg.verbose && println(">>> Train Accuracy = $(round(acc, digits = 4)), " *
                           "AUC = $(round(auc, digits = 4)), " *
                           "MCC = $(round(mcc, digits = 4))")
    cfg.verbose && println(final_named_pairs)

    voting_bias = (cld(n_top, 2) - tau) / n_top            # 多数投票下恒为 0.0

    return REOModel(cfg, final_named_pairs, voting_weights, voting_bias)
end


"""
    calc_accuracy(scores::Vector{Int16}, y::Vector{UInt8}, tau::Int)

Accuracy of the majority-vote classifier: predict 1 when `scores >= tau`.
"""
function calc_accuracy(scores::Vector{Int16}, y::Vector{UInt8}, tau::Int)
    n = length(y)
    n == 0 && return 0.0
    correct = 0
    @inbounds for i in 1:n
        pred = scores[i] >= tau ? 1 : 0
        pred == Int(y[i]) && (correct += 1)
    end
    return correct / n
end


"""
    calc_auc(scores::Vector{Int16}, y::Vector{UInt8})

Calculate the AUC for the integer voting scores scenario.
"""
function calc_auc(scores::Vector{Int16}, y::Vector{UInt8})
    pos_idx = findall(==(1), y)
    neg_idx = findall(==(0), y)
    n_pos = length(pos_idx)
    n_neg = length(neg_idx)
    (n_pos == 0 || n_neg == 0) && return 0.0
    
    # 提取正负样本的得分
    s_pos = scores[pos_idx]
    s_neg = scores[neg_idx]
    
    # 统计一致对 (Concordant pairs)
    count = 0.0
    for p in s_pos, n in s_neg
        if p > n
            count += 1.0
        elseif p == n
            count += 0.5
        end
    end
    return count / (n_pos * n_neg)
end


"""
    calc_mcc_at(scores::Vector{Int16}, y::Vector{UInt8}, tau::Int)

MCC of the majority-vote classifier at the fixed threshold `tau`.
"""
function calc_mcc_at(scores::Vector{Int16}, y::Vector{UInt8}, tau::Int)
    tp = tn = fp = fn = 0
    @inbounds for i in eachindex(scores)
        pred = scores[i] >= tau ? 1 : 0
        if y[i] == 1
            pred == 1 ? (tp += 1) : (fn += 1)
        else
            pred == 1 ? (fp += 1) : (tn += 1)
        end
    end
    denom = sqrt(Float64(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return denom == 0.0 ? 0.0 : (tp * tn - fp * fn) / denom
end
