export REOMethod, LassoMethod, RFMethod, VotingMethod, REOConfig, REOModel


"""
    REOMethod

Enumeration type for the scoring methods. It defines three possible values:

- `LassoMethod`: Lasso regression.
- `RFMethod`: Random Forest.
- `VotingMethod`: Majority Voting.
"""
@enum REOMethod LassoMethod RFMethod VotingMethod

"""
    REOConfig

Hyperparameter configuration for the REOBiomarker algorithm.

# Fields
- `method`: Scoring method — `RFMethod`, `LassoMethod`, or `VotingMethod`.
- `low_rank_q`: Percentile-rank threshold for low-expression gene filtering.
- `top_diff_n`: Number of top differentially-ranked genes to retain.
- `max_occurrence`: Maximum times a single gene may appear across candidate pairs.
- `p_val_cutoff`: p-value threshold for confounding-factor audit.
- `cor_threshold`: Correlation threshold for pruning redundant binary features.
- `target_n`: Target number of final features (RF / Lasso).
- `bqc_threshold`: Minimum enhanced-BQC score to retain a gene pair.
- `p0_threshold`: Minimum |p0 − 0.5| for control-group stability.
- `verbose`: Print diagnostic messages when `true`.
"""
Base.@kwdef struct REOConfig
    method::REOMethod = RFMethod

    # Gene-level filtering
    low_rank_q::Float64    = 0.2
    top_diff_n::Int        = 5000
    max_occurrence::Int    = 2
    p_val_cutoff::Float64  = 0.05
    cor_threshold::Float64 = 0.90

    # BQC gene-pair filtering
    bqc_threshold::Float64 = 3.0
    p0_threshold::Float64  = 0.2
	global_alpha::Union{Nothing, Float64} = nothing

    # Common selection parameters
    target_n::Int  = 15
	n_folds::Int   = 5
	metric::String = "AUC"
	seed::Int      = 43

	#Lasso specific parameters
	gamma::Float64 = 1.0

	# Vote specific parameters
	mode::String    = "soft" # soft, hard or auc
	lambda::Float64 = 0.02
	max_iter::Int   = 200

	# Printing verbosity
    verbose::Bool = false
end

"""
    REOModel

A trained REOBiomarker model holding the selected gene pairs, their weights, and bias.

# Fields
 - `config::REOConfig`: config used for training
 - `final_pairs: aligned gene pairs, `g1 > g2` is `true` in the postive group
 - `weights::Vector{Float64}`: weights for REOs
 - `intercept::Float64`: intercept (bias)
"""
struct REOModel
    config::REOConfig
    final_pairs::Vector{Tuple{String,String}}
    weights::Vector{Float64}
    intercept::Float64
end

"A trained TSP (Top Scoring Pair) model holding the selected gene pair."
struct TSPModel
    gene_i::Int
    gene_j::Int
    gene_names::Tuple{String,String}
    score::Float64
    p0::Float64  # P(Xi < Xj | Class 0)
    p1::Float64  # P(Xi < Xj | Class 1)
end

"A trained KTSP (k-Top Scoring Pair) model holding the selected gene pairs."
struct KTSPModel
    pairs::Vector{Tuple{Int,Int}}
    gene_names::Vector{Tuple{String,String}}
    scores::Vector{Float64}
    p_directions::Vector{Bool}  # true: Xi < Xj → Class 1
    k::Int
end

"A trained AUCTSP (AUC-based Top Scoring Pair) model holding the selected gene pairs."
struct AUCTSPModel
    pairs::Vector{Tuple{Int,Int}}
    gene_names::Vector{Tuple{String,String}}
    auc_scores::Vector{Float64}
    directions::Vector{Int}  # 1: Xi < Xj → Class 1; -1: Xi > Xj → Class 1
    k::Int
end
