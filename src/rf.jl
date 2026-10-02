using Random
using Statistics
using DecisionTree

"""
    _slice_ensemble(model, k)

Slice the ensemble model of DecisionTree
"""
function _slice_ensemble(model::DecisionTree.Ensemble{S, T}, k::Int) where {S, T}
    # 提取 3 个核心字段
    f1 = model.trees[1:k]    # trees
    f2 = getfield(model, 2)  # coeffs
    f3 = getfield(model, 3)  # classes
    
    # 如果对应的系数/权重向量长度与原树木数量一致，则同步进行切片截断
    if isa(f2, AbstractVector) && length(f2) == length(model.trees)
        f2 = f2[1:k]
    end
    if isa(f3, AbstractVector) && length(f3) == length(model.trees)
        f3 = f3[1:k]
    end
    
    # Re-assemble
    return DecisionTree.Ensemble{S, T}(f1, f2, f3)
end


"""
    select_model_via_cv(X, y, bqc_scores, n_folds, target_n, metric; verbose = false)

Select the best model by k-fold cross-validation.
"""
function select_model_via_cv(
    X, y, 
    bqc_scores::Vector{Float64}, 
    n_folds::Int   = 5, 
    target_n::Int  = 50,
    metric::String = "MCC"; 
	seed::Int      = 43,
    verbose::Bool  = false
)
    n_samples, n_features = size(X)
    
    idx1 = findall(==(1), y)
    idx0 = findall(==(0), y)
    
    rng = Random.MersenneTwister(seed) 
    shuffle_idx1 = idx1[randperm(rng, length(idx1))]
    shuffle_idx0 = idx0[randperm(rng, length(idx0))]
    
    folds = [Int[] for _ in 1:n_folds]
    for (i, idx) in enumerate(shuffle_idx1)
        push!(folds[mod1(i, n_folds)], idx)
    end
    for (i, idx) in enumerate(shuffle_idx0)
        push!(folds[mod1(i, n_folds)], idx)
    end
    
    # 定义候选树木颗数
    candidate_sizes = unique([1; 5:5:target_n; target_n])
    cv_scores = zeros(Float64, n_folds, length(candidate_sizes))
    
    verbose && 
	println(">>> Perform ", n_folds, " cross-validation to optimize $metric")
    
    # 2. 交叉验证循环
    for f in 1:n_folds
        val_idx   = folds[f]
        train_idx = setdiff(1:n_samples, val_idx)
        
        X_train, y_train = X[train_idx, :], y[train_idx]
        X_val, y_val     = X[val_idx, :], y[val_idx]
        
        # 在当前折的训练集上训练饱满森林
        full_fold_model = build_forest(y_train, X_train, -1, target_n, 0.7, 1)
        
        # 沿路径评估不同树木规模在验证集上的表现
        for (j, k) in enumerate(candidate_sizes)
            sub_model = _slice_ensemble(full_fold_model, k)
            #preds = apply_forest(sub_model, X_val)
			prob_matrix = apply_forest_proba(sub_model, X_val, [0, 1])
			preds = prob_matrix[:, 2]

            if metric == "AUC"
				#cv_scores[f, j] = calc_auc(Int16.(preds), UInt8.(y_val))
				cv_scores[f, j] = _binary_auc(preds, y_val)
            else
				cv_scores[f, j] = _calculate_mcc(preds .> 0.5, y_val)
            end
        end
    end
    
    # 3. 计算各个复杂度的平均验证表现
    mean_cv_scores = mean(cv_scores, dims=1)[:]
    best_idx       = argmax(mean_cv_scores)
    best_n_trees   = candidate_sizes[best_idx]
    
    verbose && println(">>> Done with CV. Optimal n_trees = $best_n_trees," *
			"average $metric  = ", round(mean_cv_scores[best_idx], digits=6))
    
    # 4. 使用最优超参数在全量数据集上构建最终模型
    verbose && println(">>> Optimize the final model with full data.")
    full_model  = build_forest(y, X, -1, target_n, 0.7, 1)
    final_model = _slice_ensemble(full_model, best_n_trees)
    
    # 5. 结合 BQC 分数计算全量模型的特征稳定性得分
    selection_scores = zeros(Float64, n_features)
    for tree in final_model.trees
        if isa(tree, Node)
            bqc_factor = log1p(bqc_scores[tree.featid])
            selection_scores[tree.featid] += 1.0 * bqc_factor
        end
    end
    
    max_s = maximum(selection_scores)
    final_scores = max_s > 0 ? selection_scores ./ max_s : selection_scores

    return final_scores, final_model
end


"""
    _fit_rf_strategy(X, bqc_scores,labels, gene_ids, pairs_idx, cfg)

Train the RF model (the main pipeline).
"""
function _fit_rf_strategy(X, bqc_scores::Vector{Float64}, labels, gene_ids, pairs_idx, cfg)
    cfg.verbose && println(">>> Train random forest (stumps) model...")

    n_folds = hasproperty(cfg, :n_folds) ? cfg.n_folds : 5
    metric  = hasproperty(cfg, :metric)  ? cfg.metric  : "AUC"
    
    # 通过 CV 筛选最佳模型
	_, final_model = select_model_via_cv(X, labels, bqc_scores, n_folds, 
						cfg.target_n, metric; seed = cfg.seed, verbose=cfg.verbose)

    # 提取最终优选模型的基尼重要性
    raw_weights = impurity_importance(final_model)

    # 4. Orientation alignment
    final_named_pairs = Vector{Tuple{String, String}}()
    aligned_weights   = Float64[]
    
	top_indices = unique([tree.featid for tree in final_model.trees if isa(tree, Node)])
	raw_weights = raw_weights[top_indices]
    
    for (i, idx) in enumerate(top_indices)
        g1_idx, g2_idx = pairs_idx[idx]
        p_rate = mean(X[labels .== 1, idx])
        n_rate = mean(X[labels .== 0, idx])
        
        if p_rate >= n_rate
            push!(final_named_pairs, (gene_ids[g1_idx], gene_ids[g2_idx]))
        else
            push!(final_named_pairs, (gene_ids[g2_idx], gene_ids[g1_idx]))
        end
        
        # 结合 BQC 调节最终决策特征的输出权重
        bqc_weight_adjusted = raw_weights[i] * log1p(bqc_scores[idx])
        push!(aligned_weights, bqc_weight_adjusted)
    end
    
    # 5. 权重归一化与截距计算
    weight_sum   = sum(aligned_weights)
    norm_weights = weight_sum > 0 ?
		aligned_weights ./ weight_sum : 
		fill(1.0 / length(aligned_weights), length(aligned_weights))
    intercept = 0
    
    return REOModel(cfg, final_named_pairs, norm_weights, intercept)
end
