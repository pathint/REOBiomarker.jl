using Distributions, QuadGK, Statistics
using Base.Threads

"""
    fit_reo_dist(data, cfg)

Fit the distribution of REOs with symmetric beta or probit-normal distributions.

Return the best fit parameter and sum of squared errors (SSE):
 ((alpha = ., sse_beta = .), (tau = ., sse_probit = .))
"""
function fit_reo_dist(data::Matrix{<:Real}, cfg::REOConfig)
    # 1. Filter low-expression genes
    cfg.verbose && println(">>> Filtering low-expression genes...")
    keep_low = filter_low_rank_genes(data, cfg.low_rank_q; verbose=cfg.verbose)

    # 2. Estimate the global alpha 
	cfg.verbose && 
	println(">>> Estimate global α (symmetric Beta) and τ (Probit-Normal) values ...")

    counts_freq = calculate_reo_distribution(data[keep_low,:]; verbose = cfg.verbose)
    counts_emp  = symmetrize_and_to_pdf(counts_freq; verbose = cfg.verbose)
    return fit_distributions(counts_emp; verbose = cfg.verbose)
end

"""
	calculate_bqc(k0, n0, k1, n1, α; n_power = 1, eps = 1e-15) -> (bqc, p0)

Calculate Bayesian Quality Control (BQC) score and posterior p0, by integrating the 
overlapping area between the posterior distributions of the REO in the control group 
and in the case group. 
`k0` and `n0` are the total number of occurence of the REO (g1 > g2) and sample size of
the control group, respectively;
`k1` and `n1` are the total number of occurence of the REO (g1 > g2) and sample size of
the case group, respectively;
`α` is the estimated parameter for the Beta prior;
`n_power` is for the penaty power (1 or 2);
`eps` is for numerical stability.
"""
function calculate_bqc(
    k0::Int, n0::Int, k1::Int, n1::Int, 
    alpha_global::Float64; 
    n_power=1, eps=1e-15
)
	# 1. Compute the posterior mean for the control group and use it
	#    as a more robust anchor for p0.
	#    Posterior mean of a Beta distribution: (α + k) / (α + β + n).
	#    The prior is symmetric, so α + β = 2 * alpha_global.
    p0_post_mean = (alpha_global + k0) / (2 * alpha_global + n0)
    
    # 2. Build the full posterior distributions for the control and case groups
    dist_control = Beta(alpha_global + k0, alpha_global + n0 - k0)
    dist_case    = Beta(alpha_global + k1, alpha_global + n1 - k1)
    
	# 3. Numerically integrate along the direction indicated by the data to obtain
	#    the base Bayesian quality score (base BQC).
    if p0_post_mean >= 0.5
		# The steady state corresponds to g1 > g2: compute P(θ_case < θ_control).
		integrand = p -> pdf(dist_control, p) * cdf(dist_case, p)
    else
		# The steady state corresponds to g1 < g2: compute P(θ_case > θ_control).
		integrand = p -> pdf(dist_control, p) * (1.0 - cdf(dist_case, p))
    end
    
	# Split the range to avoid the divergence issue at `p = 0` and `p = 1`
	base_bqc, err = quadgk(integrand, 0.0, 0.5, 1.0; rtol=1e-10)

    base_bqc = clamp(base_bqc, 0.0, 1.0)

	# 4. Transform to logarithmic space; '+eps' avoids log(0).
    log_part = -log10(1.0 - base_bqc + eps)
    
	# 5. Weight the score by how strongly the data support the observed direction,
	#    using the more robust posterior mean as the anchor.
    weight_part = (abs(p0_post_mean - 0.5) * 2.0)^n_power
    
    # 6. Combine both parts to obtain the final enhanced score.
    score = log_part * weight_part
    
    return score, p0_post_mean
end



"""
    calibrate_threshold(scores, labels) -> Float64

Find the optimal classification threshold using Youden's Index.
"""
function calibrate_threshold(scores, labels)
    thresholds = sort(unique(scores))
    length(thresholds) < 2 && return 0.5

    best_j = -1.0
    best_t = 0.5

    for t in thresholds
        preds = scores .>= t
        tp = sum((preds .== 1) .& (labels .== 1))
        tn = sum((preds .== 0) .& (labels .== 0))
        fp = sum((preds .== 1) .& (labels .== 0))
        fn = sum((preds .== 0) .& (labels .== 1))

        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1

        if j > best_j
            best_j = j
            best_t = t
        end
    end
    return best_t
end


"""
	generate_bqc_threshold_dict(n0, n1, α, bqc_threshold, p0_threshold; verbose = false) 

Generate a lookup dictionary mapping control group REO count (k0) 
to the minimum required case group count (k1) to satisfy the BQC threshold.
"""
function generate_bqc_threshold_dict(
    n0::Int,
    n1::Int,
    alpha_global::Float64,
    bqc_threshold::Float64,
    p0_threshold::Float64;
    verbose::Bool = false,
)
    # Result dictionary: k0 => k1_minimum_threshold
    threshold_dict = Dict{Int, Int}()

    verbose && println(">>> Generating threshold dictionary(p0_threshold = " * 
					   "$p0_threshold, bqc_threshold = $bqc_threshold)...")

    # Iterate through possible k0 values from 0 up to n0/2
    for k0 in 0:div(n0, 2)
        # Calculate the stable posterior mean for the control group to check p0_threshold
        p0_post_mean = (alpha_global + k0) / (2 * alpha_global + n0)
        
        # Filter based on p0_diff: if it doesn't cross the threshold, we stop searching
        # since abs(p0_post_mean - 0.5) monotonically decreases as k0 approaches n0/2
        if abs(p0_post_mean - 0.5) <= p0_threshold
            break
        end

        # The required k1 boundary is monotonic w.r.t. k0
        pre_k1 = get(threshold_dict, k0 - 1, nothing)
        fro_k1 = isnothing(pre_k1) ? ceil(Int, n1 / 2) : pre_k1
        
        for k1 in fro_k1:n1
            # Call the updated hierarchical BQC function
            score, _ = calculate_bqc(
                k0, n0, k1, n1, alpha_global; n_power = 1)
            
            if score >= bqc_threshold
                threshold_dict[k0] = k1
                threshold_dict[n0 - k0] = n1 - k1 # Enforce symmetry
                break # Once the score boundary is hit, move to the next k0
            end
        end
    end

    verbose && println("    Dictionary generation complete: retained " *
					   "$(length(threshold_dict)) valid steady-state conditions.")
    return threshold_dict
end



# ===================================================================
# 1. 计算 REO 频数分布 (多线程并行)
# ===================================================================
"""
    calculate_reo_distribution(data; verbose = false)

Calculate the REO distributions in parallel for `data`.
"""
function calculate_reo_distribution(data::Matrix{<:Real}; verbose = false)
    n_genes, n_samples = size(data)
    n_pairs = div(n_genes * (n_genes - 1), 2)
    
    verbose && println(">>> [1/4] Start to count REOs...")
    verbose && println("    $n_genes genes *  $n_samples samples, $n_pairs pairs")
   
	atomic_counts = [Threads.Atomic{Int}(0) for _ in 1:(n_samples + 1)]

	# Handle ties
	thread_rand_bits = [BitVector(undef, n_samples) for _ in 1:nthreads()]

    Threads.@threads for i in 1:(n_genes-1)
		# NOTE: dynamic assignment? `threadid()` could be larger than `nthreads()`
		id = mod1(threadid(), nthreads())
		rand_bits = thread_rand_bits[id]
        @inbounds for j in (i+1):n_genes
            k = 0
			rand!(rand_bits)
            for s in 1:n_samples
                a = data[i, s] 
				b = data[j, s]
				k += (a > b) | ((a == b) & rand_bits[s])
            end
			Threads.atomic_add!(atomic_counts[k + 1], 1)
        end
    end
    
	return [atomic_counts[i][] for i in 1:(n_samples + 1)]
end

# ===================================================================
# 2. 强制对称与经验概率密度 (Empirical PDF) 转换
# ===================================================================
"""
    symmetrize_and_to_pdf(counts, verbose = false)

Symmetrize the REO counts vectors and return emprical PDF.
"""
function symmetrize_and_to_pdf(counts::Vector{Int}; verbose = false)
    m = length(counts) - 1
    sym_counts = zeros(Float64, m + 1)
    
    verbose && println(">>> [2/4] Symmetrize and convert frequencies to PDF...")
    
    for k in 0:m
        sym_counts[k + 1] = (counts[k + 1] + counts[m - k + 1]) / 2.0
    end
    
	# Convert to the emprical PDF density, total area = 1
    dp = 1.0 / m
    emp_pdf = sym_counts ./ (sum(sym_counts) * dp)
    return emp_pdf
end

# ===================================================================
# 3. 分布拟合 (基于内部点的最小二乘法网格搜索)
# ===================================================================
"""
    fit_distributions(emp_pdf; verbose = false)

Estimate the α parameter in the Beta dsitribution and the τ parameter
in the Probit-Normal distribution.
Return the best-fit α and τ, and the fitting errors.
"""
function fit_distributions(emp_pdf::Vector{Float64}; verbose = false)
    m = length(emp_pdf) - 1
    
	# Avoid the boundary points at `p=0` and `p=1`.
    p_vals = collect(1:m-1) ./ m
    target_pdf = emp_pdf[2:end-1]
    
    verbose && println(">>> [3/4] Fit to the symmetric Beta distribution ...")
    best_alpha, min_sse_beta = 1.0, Inf
    # alpha 通常在 (0, 1] 之间表示 U型分布
    for alpha in 0.001:0.001:2.0
        pred_pdf = [pdf(Beta(alpha, alpha), p) for p in p_vals]
        sse = sum((pred_pdf .- target_pdf).^2)
        if sse < min_sse_beta
            min_sse_beta = sse
            best_alpha = alpha
        end
    end
    
	verbose && println("          Best-fit α =  $(best_alpha)")
    verbose && println(">>> [3/4] Fit to the Probit-Normal distribution ...")
    best_tau, min_sse_probit = 1.0, Inf
    norm_dist = Normal(0, 1)
    # tau 通常 > 1 表示 U型分布
    for tau in 1.01:0.01:20.0
        pred_pdf = Float64[]
        for p in p_vals
            # f(p) = (1/tau) * exp( (Phi^-1(p))^2 / 2 * (1 - 1/tau^2) )
            z = quantile(norm_dist, p)
            val = (1.0 / tau) * exp((z^2 / 2.0) * (1.0 - 1.0 / tau^2))
            push!(pred_pdf, val)
        end
        sse = sum((pred_pdf .- target_pdf).^2)
        if sse < min_sse_probit
            min_sse_probit = sse
            best_tau = tau
        end
    end
	verbose && println("          Best-fit τ =  $(best_tau)")
    
    return ((alpha=best_alpha, sse_beta=min_sse_beta), 
			(tau=best_tau, sse_probit=min_sse_probit))
end


"""
    calculate_tdi_metrics(results, bqc_threshold, p0_threshold;
	                      top_k = 50, verbose = false)
						  
Calculates the Task Difficulty Index (TDI) and classifies the dataset's 
signaling strength.
- `results`: The vector of NamedTuples containing all pairs' scores.
- `bqc_threshold`, `p0_threshold`: The filtering cutoffs.
- `top_k`: the number of top REOs used to estimate the index.
"""
function calculate_tdi_metrics(results, bqc_threshold::Float64, p0_threshold::Float64; 
		top_k::Int=50, verbose = false)
    # 1. Extract all REOs with BQC > 0
    plot_scores = [x for x in results if x.score > 0.0]
    
    # 2. Total number of REOs passing the threshold (N_effective)
    n_effective = count(x -> x.score >= bqc_threshold && x.p0_diff > p0_threshold, results)
    
    # 3. Mean BQC score for the top k REOs (mu_top)
    all_scores_sorted = sort([x.score for x in results], rev=true)
    k = min(top_k, length(all_scores_sorted))
    mu_top = k > 0 ? sum(all_scores_sorted[1:k]) / k : 0.0
    
    # 4. TDI
    tdi_score = log10(n_effective + 1) * mu_top
    
    verbose && println(">>> Evaluate task difficulity metrics ...")
    verbose && println("    Effective Feature Pool (N_effective): $n_effective")
    verbose && println("    Top-$k Core Signal Mean (mu_top): $(round(mu_top, digits=4))")
    verbose && println("    Task Difficulty Index (TDI): " *
					   "$(round(tdi_score, digits=4))")
    
    return tdi_score, plot_scores
end
