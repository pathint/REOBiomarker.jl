# src/evaluation.jl

using Random, Statistics

"""
    predict_reo(model, test_data, test_gene_ids) -> (probs, preds)

Predict class probabilities and binary labels for new samples.

For `LassoMethod` the score is passed through a sigmoid; for `RFMethod` and
`VotingMethod` the weighted vote is clamped to [0, 1].  Predictions use a
fixed threshold of 0.5.

# Arguments
- `model`: a REOModel object returned by [`fit_reo`](@ref)
- `test_data`:  gene x sample 
- `test_gene_ids`: row names for `test_data`

# Return values
`(probs, preds)`, in which `probs` are the probabilities for the samples being positive,
and `preds` are the predicted bool labels.
"""
function predict_reo(
		model::REOModel, 
		test_data::Matrix{<:Real}, 
		test_gene_ids::Vector
		)
    n_samples   = size(test_data, 2)
    gene_to_idx = Dict(name => i for (i, name) in enumerate(test_gene_ids))

    # Filter to gene pairs present in the test set
    available_weights = Float64[]
    active_pairs = []

    for (i, pair) in enumerate(model.final_pairs)
        g1_name, g2_name = pair
        if haskey(gene_to_idx, g1_name) && haskey(gene_to_idx, g2_name)
            push!(active_pairs, (gene_to_idx[g1_name], gene_to_idx[g2_name]))
            push!(available_weights, model.weights[i])
        end
    end

    isempty(active_pairs) && error("No model gene pairs found in the test set.")

    # Build binary feature matrix
    X_val = zeros(Float64, n_samples, length(active_pairs))
    for (j, (g1_idx, g2_idx)) in enumerate(active_pairs)
        #@views X_val[:, j] .= test_data[g1_idx, :] .> test_data[g2_idx, :]
        @views begin
			a = test_data[g1_idx, :]
			b = test_data[g2_idx, :]
			X_val[:, j] .=  (a .> b) .| ((a .== b) .& rand(Bool, length(a)))
		end
    end
    
    # 3. Original score `z`
    z = (X_val * available_weights) .+ model.intercept
    
    # 4. Map to probability, 0-1
    if model.config.method == LassoMethod
        # Lasso: Sigmoid (Logistic) mapping
        probs = 1.0 ./ (1.0 .+ exp.(-z))
    else
        # VotingMethod / RFMethod
        # 此时权重和为 1，z 的范围理论在 [0, 1] 之间 (假设 intercept = 0)
        # 我们将其平移回 0-1 区间
        probs = clamp.(z, 0, 1)
    end
    preds = probs .>= 0.5
    return (probs=probs, preds=preds)
end

"""
    evaluate_reo(model, valid_data, valid_gene_ids, valid_labels) -> NamedTuple

Compute performance metrics (Accuracy, MCC, AUC, and Confusion Matrix) 
for a trained REO model on a validation dataset.
"""
function evaluate_reo(
    model::REOModel,
    valid_data::Matrix{<:Real},
    valid_gene_ids::Vector,
    valid_labels::AbstractVector
)
    res = predict_reo(model, valid_data, valid_gene_ids)

    acc = mean(res.preds .== valid_labels)
    mcc = _calculate_mcc(res.preds, valid_labels)
	auc_val = _binary_auc(res.probs, valid_labels)

    tp = sum((res.preds .== 1) .& (valid_labels .== 1))
    tn = sum((res.preds .== 0) .& (valid_labels .== 0))
    fp = sum((res.preds .== 1) .& (valid_labels .== 0))
    fn = sum((res.preds .== 0) .& (valid_labels .== 1))

    if model.config.verbose
        println(">>> Performance evaluation on $(model.config.method)")
        println("    Pairs:  $(length(model.final_pairs))")
        println("    ACC:    $(round(acc, digits=4))")
        println("    MCC:    $(round(mcc, digits=4))")
        println("    AUC:    $(round(auc_val, digits=4))")
        println("    Confusion: [TP=$tp, FP=$fp; FN=$fn, TN=$tn]")
    end

    return (acc=acc, mcc=mcc, auc=auc_val, probs=res.probs, preds=res.preds)
end

"""
    _calculate_mcc(preds, labels) -> Float64

Compute the Matthews Correlation Coefficient for binary classification.
"""
function _calculate_mcc(preds::AbstractVector{Bool}, labels::AbstractVector)
    tp = Float64(sum(  preds .& (labels .== 1)))
    tn = Float64(sum(.!preds .& (labels .== 0)))
    fp = Float64(sum(  preds .& (labels .== 0)))
    fn = Float64(sum(.!preds .& (labels .== 1)))

    num = (tp * tn) - (fp * fn)
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return den == 0 ? 0.0 : num / den
end

"""
    _binary_auc(scores, labels)

Calculate the AUC for the binary classification。

`scores` and `labels` must have equal length, and `labels` must include `0` and `1`.
"""
function _binary_auc(scores::AbstractVector{<:Real}, labels::AbstractVector)
    length(scores) == length(labels) || error("scores and labels must have equal length.")

    n = length(labels)
    pos_mask = labels .== 1
    neg_mask = labels .== 0
    n_pos = count(pos_mask)
    n_neg = count(neg_mask)
    (n_pos > 0 && n_neg > 0) || error("labels must have both 0s and 1s.")

    order = sortperm(scores)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && scores[order[j + 1]] == scores[order[i]]
            j += 1
        end

        avg_rank = (i + j) / 2
        for k in i:j
            ranks[order[k]] = avg_rank
        end
        i = j + 1
    end

    pos_rank_sum = sum(ranks[pos_mask])
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
end


"""
    run_permutation_test(data, labels, genes, model, cfg; n_permutations=100)

Full permutation test: refits the model on each permuted label set.
"""
function run_permutation_test(
		data::Matrix{<:Real}, 
		labels::AbstractVector, 
		genes::Vector, 
		model::REOModel, 
		cfg::REOConfig; 
		n_permutations=100
)
    n_permutations > 0 || error("n_permutations must be a positve integer.")
    cfg.verbose && println(">>> Running permutation test ($n_permutations permutations)")

    #original_model = fit_reo(data, labels, genes, cfg)
    res = evaluate_reo(model, data, genes, labels)
    observed_mcc = res.mcc

    permuted_mccs = zeros(n_permutations)
    for i in 1:n_permutations
        shuffled_labels = shuffle(labels)
        try
            p_model = fit_reo(data, shuffled_labels, genes, cfg)
            p_res   = evaluate_reo(p_model, data, genes, shuffled_labels)
            permuted_mccs[i] = p_res.mcc
        catch
            permuted_mccs[i] = 0.0
        end
    end

	p_value = (sum(permuted_mccs .>= observed_mcc) + 1) / (n_permutations + 1)
    return (p_value=p_value, observed_mcc=observed_mcc, permuted_mccs=permuted_mccs)
end


"""
    _evaluate_binary_predictions(preds, labels; title="", verbose=false)

Evaluate the performance (accuracy, MCC and predicted vectors) for 
binary classification. Used for traditional TSP algorithms.
"""
function _evaluate_binary_predictions(
		preds::AbstractVector{Bool}, 
		scores::AbstractVector, 
		labels::AbstractVector; 
		title::String="", 
		verbose::Bool=false
		)
    acc = mean(preds .== labels)
    mcc = _calculate_mcc(preds, labels)

    tp = sum((preds .== 1) .& (labels .== 1))
    tn = sum((preds .== 0) .& (labels .== 0))
    fp = sum((preds .== 1) .& (labels .== 0))
    fn = sum((preds .== 0) .& (labels .== 1))

	if length(unique(scores)) > 2
		auc = _binary_auc(scores, labels)
	else
		auc = nothing
	end

    if verbose
        println(">>> Performace evaluation for $title")
        println("ACC: $(round(acc, digits= 6))")
		isnothing(auc) || println("AUC: $(round(auc, digits=6))")
        println("MCC) $(round(mcc, digits=6))")
        println("Confusion Matrix: [TP: $tp, FP: $fp; FN: $fn, TN: $tn]")
    end

    return (acc=acc, auc = auc, mcc=mcc, preds=preds)
end


"""
    evaluate_tsp(model, valid_data, valid_gene_ids, valid_labels; verbose=false)

Evaluate a trained TSP model on a validation dataset.
Return accuracy, MCC, and prediction result.
"""
function evaluate_tsp(
		model::TSPModel, 
		valid_data::Matrix{<:Real}, 
		valid_gene_ids::Vector, 
		valid_labels::AbstractVector; 
		verbose::Bool=false)
    return _evaluate_binary_predictions(
					predict_tsp(model, valid_data, valid_gene_ids)..., 
					valid_labels; title="TSP", verbose=verbose)
end

"""
    evaluate_ktsp(model, valid_data, valid_gene_ids, valid_labels; verbose=false)

Evaluate a trained k-TSP model on a validation dataset.
Return accuracy, AUC, MCC, and prediction result.
"""
function evaluate_ktsp(
		model::KTSPModel, 
		valid_data::Matrix{<:Real}, 
		valid_gene_ids::Vector, 
		valid_labels::AbstractVector; 
		verbose::Bool=false)
    return _evaluate_binary_predictions(
					predict_ktsp(model, valid_data, valid_gene_ids)..., 
					valid_labels; title="k-TSP", verbose=verbose)
end

"""
    evaluate_auctsp(model, valid_data, valid_gene_ids, valid_labels; verbose = false)

Evaluate a trained AUC-TSP model on a validation dataset.
Return accuracy, AUC, MCC, and prediction result.
"""
function evaluate_auctsp(
		model::AUCTSPModel, 
		valid_data::Matrix{<:Real}, 
		valid_gene_ids::Vector, 
		valid_labels::AbstractVector; 
		verbose::Bool=false)
    return _evaluate_binary_predictions(
					predict_auctsp(model, valid_data, valid_gene_ids)..., 
					valid_labels; title="AUC-TSP", verbose=verbose)
end
