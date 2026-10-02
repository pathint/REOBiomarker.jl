using Random

"""
    generate_test_data(n_genes, n_samples; rng=Random.default_rng())

Generate simulation data for test.

Return `(data, labels, genes)`：
`data` is a matrix (gene  × sample)，
`labels` is `0/1` vector,
`genes` is gene name vector.
The first two genes are picked to form reversed REO, 
and `Gene_1 > Gene_2` in the positive samples and 
`Gene_1 < Gene_2` in the negative samples.
"""
function generate_test_data(
		n_genes::Int=1000, 
		n_samples::Int=200; 
		rng=Random.default_rng())
    n_genes >= 2   || error("n_genes should be > 2.")
    n_samples >= 2 || error("n_samples should be > 2.")

    data = randn(rng, n_genes, n_samples)
    n_pos = n_samples ÷ 2
    n_neg = n_samples - n_pos
    labels = vcat(ones(Int, n_pos), zeros(Int, n_neg))

    # 构造稳定翻转的序关系，便于 BQC 和 TSP 类方法识别。
    data[1, labels .== 1] .+= 2.0
    data[2, labels .== 1] .-= 2.0
    data[1, labels .== 0] .-= 2.0
    data[2, labels .== 0] .+= 2.0
    
    genes = ["Gene_$i" for i in 1:n_genes]
    return data, labels, genes
end
