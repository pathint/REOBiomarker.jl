# src/training.jl

"""
    fit_reo(data, labels, gene_ids, cfg; confounders=nothing) -> REOModel

Train a REOBiomarker model, the main entry point for the pipeline. 
After the shared pre-filtering step, the pipeline then dispatches 
to the chosen strategy (Voting / RF / Lasso).

# Arguments
- `data`: expression profiles, gene x sample
- `labels`: sample label, `0` or `1`
- `gene_ids`: gene ids for the rows in `data`
- `cfg`: `REOConfig` configuration object
- `confounders`: optional list of confounding factors

# Return Value

Return a `REOModel` object, can be passed to [`predict_reo`](@ref) or 
[`evaluate_reo`](@ref).

# Example

```julia
cfg = REOConfig(method=VotingMethod, target_n=5)
model = fit_reo(data, labels, genes, cfg)
```
"""
function fit_reo(
    data::Matrix{<:Real},
    labels::AbstractVector,
    gene_ids::Vector,
    cfg::REOConfig;
    confounders::Union{Nothing,Vector{<:AbstractVector}} = nothing
)

    # Phase 1: shared pre-filtering
    final_pairs_idx, bqc_scores, X = 
	       preprocess_filters(data, labels, gene_ids, cfg, confounders)

    isempty(final_pairs_idx) &&
        error("No valid gene pairs after pre-filtering. Relax filtering parameters.")

    # Phase 2: method-specific training
    initial_model = if cfg.method == VotingMethod
        _fit_voting_strategy(X, bqc_scores, labels, gene_ids, final_pairs_idx, cfg)
    elseif cfg.method == RFMethod
        _fit_rf_strategy(X, bqc_scores, labels, gene_ids, final_pairs_idx, cfg)
    elseif cfg.method == LassoMethod
        _fit_lasso_strategy(X, bqc_scores, labels, gene_ids, final_pairs_idx, cfg)
    else
        error("Unknown method: $(cfg.method)")
    end

    return initial_model
end
