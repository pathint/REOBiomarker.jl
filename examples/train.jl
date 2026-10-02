using DelimitedFiles
using StatsPlots

using Pkg
Pkg.develop(path="/public/xwang/doc/icb/code/REOBiomarker/")
using REOBiomarker

gene = readdlm("gene.tsv", '\t', String, '\n')
meta = readdlm("meta.tsv", '\t', String, '\n')
expr = readdlm("expr.tsv", '\t', Float64, '\n')


genes = gene[:, 1]
label = meta[:, 2] .== "F"

# 1) Estimate task difficulty
cfg = REOConfig(
        low_rank_q = 0.0,
        bqc_threshold = 1,
        p0_threshold =  0.1,
        verbose = true)

pairs, scores, tdi_score, plot_scores = check_task_difficulty(expr, label, genes, cfg)
println("Task Diffculity Index: $tdi_score")
println("\n\n")

p0_diff    = [x.p0_diff for x in plot_scores]
bqc_scores = [x.score   for x in plot_scores]

plt = histogram(bqc_scores, xlabel = "BQC score", ylabel = "Density")
savefig(plt, "bqc_scores_histogram.pdf")

plt = histogram2d(p0_diff, bqc_scores, xlabel = "|p0 - 0.5|", ylabel = "BQC score")
savefig(plt, "bqc_scores_histogram_2d.pdf")

println("\n\n")

# 2) Basic Voting method
cfg = REOConfig(
        method = VotingMethod,
		mode   = "auc",
		max_occurrence = 10,
		cor_threshold = 1,
		target_n = 15,
        low_rank_q = 0.0,
        bqc_threshold = 4.0,
        p0_threshold =  0.3,
        verbose = true)

model  = fit_reo(expr, label, genes, cfg)
result = evaluate_reo(model, expr, genes, label)

println("Final model:")
println(model)
println("Performance on the training dataset:")
println(result)
println("\n\n")

# 3) Random Forest (stumps) method
cfg = REOConfig(
        method = RFMethod,
		target_n = 15,
        low_rank_q = 0.0,
        bqc_threshold = 4.0,
        p0_threshold =  0.3,
		metric = "AUC",
        verbose = true)

model  = fit_reo(expr, label, genes, cfg)
result = evaluate_reo(model, expr, genes, label)

println("Final model:")
println(model)
println("Performance on the training dataset:")
println(result)
println("\n\n")


# 3) Lasso method
cfg = REOConfig(
        method = LassoMethod,
		target_n = 15,
        low_rank_q = 0.0,
        bqc_threshold = 4.0,
        p0_threshold =  0.3,
        verbose = true)

model  = fit_reo(expr, label, genes, cfg)
result = evaluate_reo(model, expr, genes, label)

println("Final model:")
println(model)
println("Performance on the training dataset:")
println(result)
println("\n\n")

# 4) Traditional TSP methods

model  = fit_tsp(expr, label, genes, cfg)
result = evaluate_tsp(model, expr, genes, label)
println("Final model:")
println(model)
println("Performance on the training dataset:")
println(result)
println("\n\n")

model  = fit_ktsp(expr, label, genes, cfg; k_max = 3)
result = evaluate_ktsp(model, expr, genes, label)
println("Final model:")
println(model)
println("Performance on the training dataset:")
println(result)
println("\n\n")

model  = fit_auctsp(expr, label, genes, cfg; k_max = 3)
result = evaluate_auctsp(model, expr, genes, label)
println("Final model:")
println(model)
println("Performance on the training dataset:")
println(result)
println("\n\n")

# 5) Fit the distribution of REOs
cfg = REOConfig(
        low_rank_q = 0.0,
        verbose = true)

beta_res, probit_res = fit_reo_dist(expr, cfg)
println("Beta fit result: $beta_res")
println("Probit-Normal fit result: $probit_res")
