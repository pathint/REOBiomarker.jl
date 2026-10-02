# src/filters.jl

using Statistics, StatsBase, HypothesisTests, Combinatorics

"""
    preprocess_filters(data, labels, gene_ids, cfg, confounders=nothing) 

Run the full pre-filtering pipeline: low-expression removal, differential
ranking, BQC pair filtering, optional confounding-factor audit, hub-gene
pruning, feature-matrix construction, and correlation pruning.
Return the selected gene pairs and the REO matrix, `(final_pairs, X)`
"""
function preprocess_filters(
    data::Matrix{<:Real},
    labels::AbstractVector,
    gene_ids::Vector,
    cfg::REOConfig,
    confounders::Union{Nothing,Vector{<:AbstractVector}}=nothing,
)
    # 1. Filter low-expression genes
    cfg.verbose && println(">>> Filtering low-expression genes...")
    keep_low = filter_low_rank_genes(data, cfg.low_rank_q; verbose=cfg.verbose)

    # 2. Differential rank filter
    cfg.verbose && println(">>> Filtering by differential rank...")
    selected_genes = filter_diff_rank_genes(
        data, labels, keep_low; top_n=cfg.top_diff_n, verbose=cfg.verbose)

    # 3. Generate all candidate gene pairs
    all_pairs = collect(combinations(selected_genes, 2))
    cfg.verbose && println(">>> Initial $(length(all_pairs)) candidate pairs generated.")

    # 3.1 BQC filtering
	pairs_initial, bqc_scores, _, _ = filter_pairs_by_bqc(
								      all_pairs, data, labels, keep_low, cfg)

    cfg.verbose &&
        println(">>> After BQC filtering: $(length(pairs_initial)) pairs remain.")

    isempty(pairs_initial) && 
	    error("No gene pairs remain after BQC. Relax `bqc_threshold` or `p0_threshold`.")

    # 4. Confounding-factor audit
    if !isnothing(confounders) && !isempty(confounders)
        cfg.verbose && println(">>> Auditing against $(length(confounders)) confounders...")

        keep_mask = fill(true, length(pairs_initial))
        p_val_cutoff = cfg.p_val_cutoff

        for (i, pair) in enumerate(pairs_initial)
            g1_idx, g2_idx = pair
            reo_vec = data[g1_idx, :] .> data[g2_idx, :]

            for cf_vec in confounders
                if is_confounded(reo_vec, cf_vec, p_val_cutoff)
                    keep_mask[i] = false
                    break
                end
            end
        end

        n_removed = count(!, keep_mask)
        pairs_initial = pairs_initial[keep_mask]
        cfg.verbose && println("    Removed $n_removed pairs correlated with covariates.")
    end

    # 5. Hub-gene pruning
    pairs_idx    = prune_hub_genes(pairs_initial, cfg.max_occurrence; verbose=cfg.verbose)
	pairs_pruned = pairs_initial[pairs_idx]
	bqc_scores   = bqc_scores[pairs_idx]

    # 6. Build direction-aligned binary feature matrix
    X_initial, pairs_pruned = build_feature_matrix_aligned(data, pairs_pruned, labels)

    # 7. Correlation pruning
    keep = drop_correlated_features(
        X_initial, pairs_pruned, cfg.cor_threshold
    )
	cfg.verbose && 
	println(">>> $(length(keep)) pairs are kept after removing correlated features")
	final_pairs = pairs_pruned[keep]
	bqc_scores  = bqc_scores[keep]
	X_final     = X_initial[:, keep]
    return final_pairs, bqc_scores, X_final
end

"""
    filter_low_rank_genes(data, threshold=0.2; verbose=false)

Remove genes whose median within-sample percentile rank falls below `threshold`.
Return the gene index vector for the kept ones.
"""
function filter_low_rank_genes(data::Matrix{<:Real}, threshold=0.2; verbose=false)
    n_genes, n_samples = size(data)
    percentile_ranks = Matrix{Float64}(undef, n_genes, n_samples)
    Threads.@threads for j in 1:n_samples
        percentile_ranks[:, j] .= tiedrank(data[:, j]) ./ n_genes
    end

    keep_indices = findall(i -> median(percentile_ranks[i, :]) > threshold, 1:n_genes)
    verbose && println("    Remove low-expression genes," *
				" kept $(length(keep_indices)) / $n_genes genes.")
    return keep_indices
end

"""
    filter_diff_rank_genes(data, labels, gene_indices; top_n=500, verbose=false)

Retain the `top_n` genes with the largest absolute mean percentile-rank
difference between the two classes.
Return the gene index vector for the kept ones.
"""
function filter_diff_rank_genes(
    data::Matrix{<:Real},
    labels::AbstractVector,
    gene_indices::Vector{Int};
    top_n=500,
    verbose=false,
)
    n_samples = size(data, 2)
    n_genes_subset = length(gene_indices)

    sub_data = data[gene_indices, :]
    percentile_ranks = Matrix{Float64}(undef, n_genes_subset, n_samples)
    for j in 1:n_samples
        percentile_ranks[:, j] .= tiedrank(sub_data[:, j]) ./ n_genes_subset
    end

    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)

    # Absolute mean difference
    diffs = [
        abs(mean(percentile_ranks[i, idx1]) - mean(percentile_ranks[i, idx0])) 
		for i in 1:n_genes_subset
    ]

    p = sortperm(diffs; rev=true)
    selected_internal_indices = p[1:min(top_n, length(p))]

    final_indices = gene_indices[selected_internal_indices]
    verbose && println("    Remove low mean-rank difference genes, " *
					   "kept $(length(final_indices)) genes.")
    return final_indices
end

"""
    get_top_pairs_parallel_fisher(data, labels, gene_indices; n_top=5000, verbose=false)

Parallel Fisher exact test to rank gene pairs by discriminative power.
"""
function get_top_pairs_parallel_fisher(
    data::Matrix{<:Real},
    labels::AbstractVector,
    gene_indices::Vector{Int};
    n_top=5000,
    verbose=false,
)
    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0, n1 = length(idx0), length(idx1)

    all_pairs = collect(combinations(gene_indices, 2))
    n_pairs = length(all_pairs)
    p_values = Vector{Float64}(undef, n_pairs)

    # Fisher test for each pair
    Threads.@threads for i in 1:n_pairs
        g1, g2 = all_pairs[i]
        # 统计在 Label=1 中 g1 > g2 的数量
		a = @view data[g1, idx1]
		b = @view data[g2, idx1]
        #c1 = sum(data[g1, idx1] .> data[g2, idx1])
		c1 = sum((a .> b) .| ((a .== b) .& rand(Bool, length(a))))
        # 统计在 Label=0 中 g1 > g2 的数量
		a = @view data[g1, idx0]
		b = @view data[g2, idx0]
        #c0 = sum(data[g1, idx0] .> data[g2, idx0])
		c0 = sum((a .> b) .| ((a .== b) .& rand(Bool, length(a))))
        # 构建 2x2 混淆矩阵
        #          g1>g2  g1<=g2
        # Label 1:  c1    n1-c1
        # Label 0:  c0    n0-c0
        ft = FisherExactTest(c1, n1-c1, c0, n0-c0)
        p_values[i] = pvalue(ft)
    end

    sp = sortperm(p_values)
    top_indices = sp[1:min(n_top, n_pairs)]

    verbose && println("  Fisher exact test: retained $(min(n_top, n_pairs)) pairs.")
    return all_pairs[top_indices], p_values[top_indices]
end


function filter_pairs_by_bqc(pairs, data, labels, keep_low, cfg::REOConfig)
    idx0 = findall(==(0), labels) # control
    idx1 = findall(==(1), labels) # case
    n0 = length(idx0)
    n1 = length(idx1)
    
	if isnothing(cfg.global_alpha)
    	# 2. Estimate the global alpha 
    	cfg.verbose && println(">>> Estimate the global alpha value ...")
    	counts_freq = calculate_reo_distribution(data[keep_low,:]; verbose = cfg.verbose)
    	counts_emp  = symmetrize_and_to_pdf(counts_freq; verbose = cfg.verbose)
    	beta_res, probit_res = fit_distributions(counts_emp; verbose = cfg.verbose)
    	alpha_global = beta_res.alpha 
	else
		alpha_global = cfg.global_alpha
	end

    bqc_threshold = cfg.bqc_threshold
    p0_threshold  = cfg.p0_threshold

    threshold_dict = generate_bqc_threshold_dict(n0, n1, alpha_global, 
												 bqc_threshold, p0_threshold; 
												 verbose=cfg.verbose)
    cfg.verbose && 
	println(">>> Filtering $(length(threshold_dict)) records in precomputed dictinoary.")
    pairs_initial = filter_pairs_with_dict(pairs, data, labels, threshold_dict; 
										   verbose = cfg.verbose)

    n_pairs = length(pairs_initial)
    results = Vector{NamedTuple{(:pair, :score, :p0, :p1, :p0_post, :p0_diff), 
								Tuple{Tuple{Int, Int}, 
									  Float64, Float64, Float64, Float64, Float64}
								}}(undef, n_pairs)
    
    cfg.verbose && println(">>> Start to calcualte the BQC score ...")

	# 3. Estimate BQC for each gene pair
    Threads.@threads for i in 1:n_pairs
        g1, g2 = pairs_initial[i]
        
		# Fast computation of REOs
        @views begin
			a = data[g1, idx0]
			b = data[g2, idx0]
			k0 = sum((a .> b) .| ((a .== b) .& rand(Bool, length(a))))
		end

        @views begin
			a = data[g1, idx1]
			b = data[g2, idx1]
			k1 = sum((a .> b) .| ((a .== b) .& rand(Bool, length(a))))
		end
        
        score, p0_post_mean = calculate_bqc(
            k0, n0, k1, n1, alpha_global; n_power=1)
        
        p0_diff = abs(p0_post_mean - 0.5)
        results[i] = (pair = (g1, g2), score = score, p0 = k0/n0, p1 = k1/n1, 
					  p0_post = p0_post_mean, p0_diff = p0_diff)
    end

	# 4. Calculate task difficulity metric
	tdi_score, plot_scores = calculate_tdi_metrics(results, bqc_threshold, p0_threshold; 
												   top_k=50, verbose = cfg.verbose)
	
	# 5. Filter gene pairs: BQC score（>=）and p0_diff（>)
    valid_results = filter(x -> x.score >= bqc_threshold && x.p0_diff > p0_threshold, 
						   results)
    #    Sort in-place, first by score, then by p0_diff
    sort!(valid_results, by = x -> (x.score, x.p0_diff), rev = true)

    cfg.verbose && println(">>> Done with BQC filtering. Total # pairs: $n_pairs, " *
						   "retained: $(length(valid_results))")

	return ([x.pair for x in valid_results], [x.score for x in valid_results], 
			tdi_score, plot_scores)
end


"""
    filter_pairs_with_dict(pairs, data, labels, threshold_dict; verbose = false)

Peform a fast gene pairs filtering step with a pre-calculated dictionary of thresholds.
"""
function filter_pairs_with_dict(pairs, data, labels, threshold_dict::Dict; verbose=false)
    idx0 = findall(==(0), labels)
    idx1 = findall(==(1), labels)
    n0 = length(idx0) / 2

    keep_mask = fill(false, length(pairs))

    Threads.@threads for i in 1:length(pairs)
        g1, g2 = pairs[i]
        k0 = sum(data[g1, idx0] .> data[g2, idx0])

        limit = get(threshold_dict, k0, nothing)

        if !isnothing(limit)
            k1 = sum(data[g1, idx1] .> data[g2, idx1])
            if (k0 < n0 && k1 >= limit) || (k0 > n0 && k1 <= limit)
                keep_mask[i] = true
            end
        end
    end

    verbose && println("    BQC dictionary filter: retained $(sum(keep_mask)) pairs.")
    return pairs[keep_mask]
end

"""
    prune_hub_genes(pairs, max_occurrence=2; verbose=false)

Prevent any single gene from appearing in more than `max_occurrence` pairs.
Return the index vector for the kept gene pairs.
"""
function prune_hub_genes(pairs, max_occurrence=2; verbose=false)
    gene_counts = Dict{Int,Int}()
    final_idx   = []

    #for (g1, g2) in pairs
	for i in 1:length(pairs)
		g1, g2 = pairs[i]
        c1 = get(gene_counts, g1, 0)
        c2 = get(gene_counts, g2, 0)
        
        if c1 < max_occurrence && c2 < max_occurrence
            #push!(final_pairs, (g1, g2))
            push!(final_idx, i)
            gene_counts[g1] = c1 + 1
            gene_counts[g2] = c2 + 1
        end
    end
	verbose && println("    After pruning: $(length(final_idx)) gene-pairs are kept.")
    return final_idx
end


"""
    build_feature_matrix_aligned(data, pairs, labels) -> (X, new_pairs)

Build a binary feature matrix with direction aligned so that `g1 > g2`
correlates with the positive class.  Returns the matrix and the (possibly
flipped) pair indices.
"""
function build_feature_matrix_aligned(data::Matrix{<:Real}, pairs, labels::AbstractVector)
    n_samples = size(data, 2)
    n_pairs = length(pairs)

    X = BitArray(undef, (n_samples, n_pairs))
    new_pairs = Vector{Tuple{Int,Int}}(undef, n_pairs)

    pos_idx = findall(==(1), labels)
    neg_idx = findall(==(0), labels)
    n_pos = length(pos_idx)
    n_neg = length(neg_idx)

    Threads.@threads for j in 1:n_pairs
        g1, g2 = pairs[j]

        #@views p_pos = sum(data[g1, pos_idx] .> data[g2, pos_idx]) / n_pos
        #@views p_neg = sum(data[g1, neg_idx] .> data[g2, neg_idx]) / n_neg

        @views begin
			a = data[g1, pos_idx]
			b = data[g2, pos_idx]
			p_pos = sum((a .> b) .| ((a .== b) .& rand(Bool, length(a)))) / n_pos
		end
        @views begin
			a = data[g1, neg_idx]
			b = data[g2, neg_idx]
			p_neg = sum((a .> b) .| ((a .== b) .& rand(Bool, length(a)))) / n_neg
		end

        # 如果 g1 > g2 在正类中出现的概率更低，则翻转这对基因
        if p_pos < p_neg
            actual_g1, actual_g2 = g2, g1
            new_pairs[j] = (g2, g1)
        else
            actual_g1, actual_g2 = g1, g2
            new_pairs[j] = (g1, g2)
        end

        # 填充特征矩阵
        #@views X[:, j] .= data[actual_g1, :] .> data[actual_g2, :]
        @views begin
			a = data[actual_g1, :]
			b = data[actual_g2, :]
			X[:, j] .=  (a .> b) .| ((a .== b) .& rand(Bool, length(a)))
		end
    end

    return X, new_pairs
end

"""
    drop_correlated_features(X, pairs, threshold=0.95) -> (pairs, X)

Remove features whose pairwise Pearson correlation exceeds `threshold`.
"""
function drop_correlated_features(X::AbstractMatrix, pairs, threshold=0.95)
    n_features = size(X, 2)
    keep = trues(n_features)
    cor_mat = cor(X)

    for i in 1:n_features
        !keep[i] && continue
        for j in (i + 1):n_features
            if keep[j] && abs(cor_mat[i, j]) > threshold
                keep[j] = false
            end
        end
    end

    return keep
end

"""
    is_confounded(reo_vec, cf_vec, p_threshold) -> Bool

Test whether a gene-pair ordering is significantly associated with a
confounding variable (continuous → Welch t-test, categorical → chi-squared).
"""
function is_confounded(
		reo_vec::AbstractVector{Bool}, 
		cf_vec::AbstractVector, 
		p_threshold::Float64)

    if eltype(cf_vec) <: AbstractFloat
        # Welch's T-test for continous factor, e.g. age
        group0 = cf_vec[reo_vec .== 0]
        group1 = cf_vec[reo_vec .== 1]

        length(group0) < 5 || length(group1) < 5 && return false
        std(group0) ≈ 0 && std(group1) ≈ 0 && return false

        return pvalue(UnequalVarianceTTest(group0, group1)) < p_threshold
    else
        # Fisher's exact test for categorical factor, e.g. Sex, Batch.
        #tbl = counts(reo_vec, cf_vec)
        categories = unique(cf_vec)
		length(categories) > 1 || return false

        cf_to_col = Dict{Any, Int}(category => i for (i, category) in enumerate(categories))
		tbl = zeros(Int, 2, length(categories))
		for (reo, cf) in zip(reo_vec, cf_vec)
			tbl[reo ? 2 : 1, cf_to_col[cf]] += 1
		end
		try
            return pvalue(ChisqTest(tbl)) < p_threshold
        catch
            return false
        end
    end
end

"""
    filter_genes(data, labels, gene_ids, cfg) -> Vector{Int}

Gene-level pre-filtering for TSP-family methods (low-expression + differential
rank filtering only, no BQC).
"""
function filter_genes(
    data::Matrix{<:Real}, labels::AbstractVector, gene_ids::Vector, cfg::REOConfig
)
    keep_low = filter_low_rank_genes(data, cfg.low_rank_q; verbose=cfg.verbose)
    selected_genes = filter_diff_rank_genes(
        data, labels, keep_low; top_n=cfg.top_diff_n, verbose=cfg.verbose
    )
    return selected_genes
end

"""
    check_task_difficulty(data, labels, gene_ids, cfg)

Evaluate the classification task difficulty by following the same train pipeline.
"""
function check_task_difficulty(
    data::Matrix{<:Real}, 
    labels::AbstractVector, 
    gene_ids::Vector, 
    cfg::REOConfig 
)
    # Same preprocess steps as `preprocess_filters`
	# 1. Filter low-expressed genes 
    keep_low = filter_low_rank_genes(data, cfg.low_rank_q; verbose=cfg.verbose)
    # 2. Filter by differential ranks
    selected_genes = filter_diff_rank_genes(data, labels, keep_low;
                                            top_n=cfg.top_diff_n, verbose=cfg.verbose)
    # 3. Generate all gene pairs
    all_pairs = collect(combinations(selected_genes, 2))

    return filter_pairs_by_bqc(all_pairs, data, labels, keep_low, cfg)
	#return pairs, bqc_scores, tdi_score, plot_scores
end
