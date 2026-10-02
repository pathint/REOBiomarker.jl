using GLMNet
using Statistics


# ==============================================================================
# Lasso 训练策略：LassoPath 自动选择
# ==============================================================================
"""
    _fit_lasso_strategy(X, bqc_scores, labels, gene_ids, pairs_idx, cfg)

Train pipeline for the Lasso model.
"""
function _fit_lasso_strategy(X, bqc_scores, labels, gene_ids, pairs_idx, cfg)
    cfg.verbose && println(">>> Perform Lasso-based gene-pairs selection ...")
    
	indices, weights, intercept = bqc_guided_lasso_cv(UInt8.(X), UInt8.(labels), bqc_scores;
												 gamma   = cfg.gamma,
												 n_folds = cfg.n_folds,
												 seed    = cfg.seed,
												 verbose = cfg.verbose)
    
    # 正向化对齐 (Orientation Alignment)
    # 目标：调整基因对顺序使得所有权重均为正，支持 Positive 类
    final_named_pairs = Vector{Tuple{String, String}}()
	for i in indices
		g1, g2 = pairs_idx[i]
		g1_name, g2_name = gene_ids[g1], gene_ids[g2]
		push!(final_named_pairs, (g1_name, g2_name))
	end
    
    isempty(weights) && 
	error("Lasso does not find any gene-pairs. Relax the selection threholds.")
    
    return REOModel(cfg, final_named_pairs, weights, intercept)
end

"""
    bqc_guided_lasso_cv(X, y, bqc_scores; gamma = 1.0, n_folds = 5, seed = 43)

The Lasso model training method guided by BQC.

# Arguments
- X: REO feature matrix (0/1 value) (row: sample, column: feature)
- y: Label (0/1)
- bqc_scores: BQC for each feature
- gamma: penality size (larger gamma, larger penality for low BQC features)
"""
function bqc_guided_lasso_cv(
    X::Matrix{UInt8}, 
    y::Vector{UInt8}, 
    bqc_scores::Vector{Float64};
    gamma::Float64 = 1.0,   # 惩罚放大杠杆
    n_folds::Int = 5,       # 交叉验证折数
    seed::Int = 43,
	verbose::Bool = false
)
    n, m = size(X)
    length(bqc_scores) == m || 
	error("bqc_scores vector must match with the column size of X")

    # 1. 将二值矩阵转换为 GLMNet 要求的 Float64 格式
    X_double = Float64.(X)

	# 第一列是负类计数 (Control组, 当 y=0 时为 1)，
	# 第二列是正类计数 (Case组,    当 y=1 时为 1)
	y_double = Float64.(y)
	y_matrix = hcat(1.0 .- y_double, y_double)

    # 2. 将 BQC 分数映射为自适应 Lasso 惩罚因子 (Adaptive Weights)
    # BQC 越大 -> 惩罚因子越小 -> 越容易被 Lasso 保留
    # 采用 log1p 进行平滑，防止个别极大 BQC 值造成惩罚因子断层
    raw_weights = 1.0 ./ ((log1p.(bqc_scores)) .^ gamma .+ 1e-5)
    
    # 归一化权重：使权重的均值为 1.0。
    penalty_factors = raw_weights ./ mean(raw_weights)

    # 3. 调用带 penalty_factor 的分层交叉验证 LassoPath
    # penalty_factor = 0 表示绝不剔除，值越大越容易被剔除
    verbose && println(">>> Running BQC-guided adaptive Lasso with $n_folds-fold CV...")
    cv_result = glmnetcv(
        X_double, y_matrix, 
        Binomial(),                  # 二分类 Logistic Lasso
        penalty_factor = penalty_factors, 
        nfolds = n_folds
    )

    # 4. 提取交叉验证表现最好（Loss 最小）的 Lambda 对应的系数
	best_lambda_idx   = argmin(cv_result.meanloss)
    best_coefficients = cv_result.path.betas[:, best_lambda_idx]
    intercept         = cv_result.path.a0[best_lambda_idx]

    # 5. 筛选出系数不为0的特征，即最终选中的基因对
    selected_indices = findall(!=(0.0), best_coefficients)
    selected_weights = best_coefficients[selected_indices]

    verbose && println(">>> Lasso Optimization Report")
    verbose && println("    Active Features Selected: $(length(selected_indices))")
    verbose && println("    Selected Feature Indices: $selected_indices")
	verbose && println("    Selected Feature Weights: " * 
					   "$(round.(selected_weights, digits=4))")
    verbose && println("    Intercept: $(round(intercept, digits=4))")

    # 返回选中特征的索引，以及对应的权重系数
    return selected_indices, selected_weights, intercept
end
