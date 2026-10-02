module REOBiomarker

export REOConfig, REOModel
export fit_reo, predict_reo, evaluate_reo
export run_permutation_test, generate_test_data
export check_task_difficulty
export fit_reo_dist

# Traditional TSP baselines
export TSPModel, KTSPModel, AUCTSPModel
export fit_tsp, predict_tsp, fit_ktsp, predict_ktsp, fit_auctsp, predict_auctsp
export evaluate_tsp, evaluate_ktsp, evaluate_auctsp

include("types.jl")       # Type definitions
include("filters.jl")     # Gene and gene-pair filtering
include("training.jl")    # Stability selection and model fitting
include("vote.jl")        # Majority voting feature subset search
include("lasso.jl")       # Weighted model, trained with Lasso
include("rf.jl")          # Weighted model, random stumps
include("evaluation.jl")  # Prediction, evaluation, and permutation tests
include("utils.jl")       # Test data generation utilities
include("statistics.jl")  # Bayesian quality control and tau/alpha estimation
include("tsp.jl")         # Classical Top Scoring Pair
include("ktsp.jl")        # Classical k-Top Scoring Pairs
include("auctsp.jl")      # Classical AUC-based Top Scoring Pairs

end
