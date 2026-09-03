using Random
using Statistics

# =========================================================
# Gray code + exact counting majority vote feature subset
# Correct + fast + brute-force consistent
# =========================================================

# -------------------------
# Feature representation
# -------------------------
struct BitFeature
    idx::Vector{Int}
end

# -------------------------
# Build features (dense → sparse index form)
# -------------------------
function build_features(X::Matrix{UInt8})
    n, m = size(X)
    feats = Vector{BitFeature}(undef, m)

    for j in 1:m
        idx = Int[]
        for i in 1:n
            if X[i, j] == 1
                push!(idx, i)
            end
        end
        feats[j] = BitFeature(idx)
    end

    return feats
end

# -------------------------
# Feature sorting (important for pruning)
# -------------------------
function sort_features!(feats, y)
    m = length(feats)
    score = zeros(Float64, m)

    @inbounds for j in 1:m
        s = 0
        for i in feats[j].idx
            s += (y[i] == 1 ? 1 : -1)
        end
        score[j] = s
    end

    perm = sortperm(score, rev=true)
    return feats[perm]
end

# -------------------------
# Add / remove feature (exact update)
# -------------------------
function add_feature!(counts::Vector{Int16}, feat::BitFeature)
    @inbounds for i in feat.idx
        counts[i] += 1
    end
end

function remove_feature!(counts::Vector{Int16}, feat::BitFeature)
    @inbounds for i in feat.idx
        counts[i] -= 1
    end
end

# -------------------------
# Classification error
# -------------------------
function compute_error(counts::Vector{Int16}, k::Int, y::Vector{UInt8})
    n = length(y)
    thresh = (k + 1) >>> 1

    err = 0
    @inbounds for i in 1:n
        pred = counts[i] >= thresh ? 1 : 0
        err += (pred != y[i])
    end

    return err
end

# -------------------------
# Main Gray code search
# -------------------------
function gray_search(feats, y)

    n = length(y)
    m = length(feats)

    counts = zeros(Int16, n)

    best_err = typemax(Int)
    best_mask = UInt64(0)

    prev_gray = UInt64(0)
    k = 0

    total = UInt64(1) << m

    @inbounds for t in UInt64(0):(total - 1)

        g = t ⊻ (t >> 1)

        if t > 0
            diff = g ⊻ prev_gray
            j = trailing_zeros(diff) + 1

            if ((g >> (j - 1)) & 1) == 1
                add_feature!(counts, feats[j])
                k += 1
            else
                remove_feature!(counts, feats[j])
                k -= 1
            end
        end

        err = compute_error(counts, k, y)

        if err < best_err || (err == best_err && k < count_ones(best_mask))
            best_err = err
            best_mask = g

            if best_err == 0
                break
            end
        end

        prev_gray = g
    end

    return best_mask, best_err
end


function compute_error_learnable(counts::Vector{Int16}, k::Int, y::Vector{UInt8})

    n = length(y)

    hist_pos = zeros(Int, k+1)
    hist_neg = zeros(Int, k+1)

    @inbounds for i in 1:n
        c = counts[i] + 1
        if y[i] == 1
            hist_pos[c] += 1
        else
            hist_neg[c] += 1
        end
    end

    cum_pos = cumsum(hist_pos)
    cum_neg = cumsum(hist_neg)

    total_pos = cum_pos[end]
    total_neg = cum_neg[end]

    best_err = typemax(Int)
    best_tau = 0

    @inbounds for τ in 0:k
        fn = τ == 0 ? 0 : cum_pos[τ]
        fp = total_neg - (τ == 0 ? 0 : cum_neg[τ])
        err = fn + fp

        if err < best_err
            best_err = err
            best_tau = τ
        end
    end

    return best_err, best_tau
end

function gray_search_learnable(feats, y)

    n = length(y)
    m = length(feats)

    counts = zeros(Int16, n)

    best_err = typemax(Int)
    best_mask = UInt64(0)
    best_tau = 0

    prev_gray = UInt64(0)
    k = 0

    total = UInt64(1) << m

    n_pos = sum(y)
    n_neg = n - n_pos
    baseline_err = min(n_pos, n_neg)

    best_err = baseline_err

    @inbounds for t in UInt64(0):(total - 1)

        g = t ⊻ (t >> 1)

        if t > 0
            diff = g ⊻ prev_gray
            j = trailing_zeros(diff) + 1

            if ((g >> (j - 1)) & 1) == 1
                add_feature!(counts, feats[j])
                k += 1
            else
                remove_feature!(counts, feats[j])
                k -= 1
            end
        end

        # Skip empty set
        if k == 0
            prev_gray = g
            continue
        end

        err, tau = compute_error_learnable(counts, k, y)

        if err < best_err || (err == best_err && k < count_ones(best_mask))
            best_err = err
            best_mask = g
            best_tau = tau

            if best_err == 0
                break
            end
        end

        prev_gray = g
    end

    return best_mask, best_err, best_tau
end


# -------------------------
# Brute force verifier (for small m only)
# -------------------------
function brute_force(X, y)
    n, m = size(X)
    best_err = n
    best_mask = 0

    for mask in 0:(1 << m) - 1
        counts = zeros(Int, n)
        k = 0

        for j in 1:m
            if (mask >> (j - 1)) & 1 == 1
                k += 1
                for i in 1:n
                    counts[i] += X[i, j]
                end
            end
        end

        thresh = (k + 1) >>> 1
        err = 0

        for i in 1:n
            pred = counts[i] >= thresh ? 1 : 0
            err += (pred != y[i])
        end

        if err < best_err
            best_err = err
            best_mask = mask
        end
    end

    return best_mask, best_err
end

function mask_to_indices(mask::Union{UInt64, UInt128}, m::Int)
    idx = Int[]
    for j in 1:m
        if (mask >> (j - 1)) & 1 == 1
            push!(idx, j)
        end
    end
    return idx
end


# --------------------------------------------
# Evaluate error for a fixed subset mask
# --------------------------------------------
function eval_mask_error(mask::Union{UInt64, UInt128}, X::Matrix{UInt8}, y::Vector{UInt8})
    n, m = size(X)

    counts = zeros(Int16, n)
    k = 0

    for j in 1:m
        if (mask >> (j - 1)) & 1 == 1
            k += 1
            @inbounds for i in 1:n
                counts[i] += X[i, j]
            end
        end
    end

    thresh = (k + 1) >>> 1

    err = 0
    @inbounds for i in 1:n
        pred = counts[i] >= thresh ? 1 : 0
        err += (pred != y[i])
    end

    return err
end

function permutation_test_pvalue(mask::Union{UInt64, UInt128},
                                 X::Matrix{UInt8},
                                 y::Vector{UInt8};
                                 B::Int = 1000,
                                 seed::Int = 1)

    rng = MersenneTwister(seed)

    obs_err = eval_mask_error(mask, X, y)

    n = length(y)
    perm_errs = zeros(Int, B)

    @inbounds for b in 1:B
        y_perm = copy(y)
        shuffle!(rng, y_perm)
        perm_errs[b] = eval_mask_error(mask, X, y_perm)
    end

    count = 0
    @inbounds for b in 1:B
        if perm_errs[b] <= obs_err && obs_err > 0
            count += 1
        end
    end

    p_value = (count + 1) / (B + 1)

    return (
        observed_error = obs_err,
        p_value = p_value,
        perm_errors = perm_errs
    )
end


function add_bit(mask::UInt64, j::Int)
    return mask | (UInt64(1) << (j-1))
end

function remove_bit(mask::UInt64, j::Int)
    return mask & ~(UInt64(1) << (j-1))
end

function add_bit(mask::UInt128, j::Int)
    return mask | (UInt128(1) << (j-1))
end

function remove_bit(mask::UInt128, j::Int)
    return mask & ~(UInt128(1) << (j-1))
end

function has_bit(mask::Union{UInt64, UInt128}, j::Int)
    return (mask >> (j-1)) & 1 == 1
end

function sffs_search(X::Matrix{UInt8}, y::Vector{UInt8};
                     max_k::Int=typemax(Int),
                     max_iter::Int=1000)

    n, m = size(X)

    current_mask = UInt128(0)
    current_err = eval_mask_error(current_mask, X, y)

    best_mask = current_mask
    best_err = current_err

    iter = 0

    while iter < max_iter
        iter += 1

        improved = false

        # Step 1: Forward inclusion
        best_add_err = current_err
        best_add_j = 0

        for j in 1:m
            if !has_bit(current_mask, j)
                new_mask = add_bit(current_mask, j)
                err = eval_mask_error(new_mask, X, y)

                if err < best_add_err
                    best_add_err = err
                    best_add_j = j
                end
            end
        end

        if best_add_j != 0
            current_mask = add_bit(current_mask, best_add_j)
            current_err = best_add_err
            improved = true
        else
            break
        end

        # Step 2: Conditional backward
        while true
            best_remove_err = current_err
            best_remove_j = 0

            for j in 1:m
                if has_bit(current_mask, j)
                    new_mask = remove_bit(current_mask, j)
                    err = eval_mask_error(new_mask, X, y)

                    if err < best_remove_err
                        best_remove_err = err
                        best_remove_j = j
                    end
                end
            end

            if best_remove_j != 0
                current_mask = remove_bit(current_mask, best_remove_j)
                current_err = best_remove_err
                improved = true
            else
                break
            end
        end

        # Track global best
        if current_err < best_err ||
           (current_err == best_err &&
            count_ones(current_mask) < count_ones(best_mask))

            best_mask = current_mask
            best_err = current_err
        end

        if count_ones(current_mask) >= max_k
            break
        end

        if !improved
            break
        end
    end

    return best_mask, best_err
end

function compute_error_weighted(counts::Vector{Int16},
                                k::Int,
                                y::Vector{UInt8};
                                w_pos::Float64=1.0,
                                w_neg::Float64=1.0)

    n = length(y)

    hist_pos = zeros(Int, k+1)
    hist_neg = zeros(Int, k+1)

    @inbounds for i in 1:n
        c = counts[i] + 1
        if y[i] == 1
            hist_pos[c] += 1
        else
            hist_neg[c] += 1
        end
    end

    cum_pos = cumsum(hist_pos)
    cum_neg = cumsum(hist_neg)

    total_pos = cum_pos[end]
    total_neg = cum_neg[end]

    best_err = Inf
    best_tau = 0

    @inbounds for τ in 0:k
        fn = τ == 0 ? 0 : cum_pos[τ]
        fp = total_neg - (τ == 0 ? 0 : cum_neg[τ])

        err = w_pos * fn + w_neg * fp

        if err < best_err
            best_err = err
            best_tau = τ
        end
    end

    return best_err, best_tau
end

function compute_class_weights(y)
    n = length(y)
    n_pos = sum(y)
    n_neg = n - n_pos

    w_pos = n / (2 * max(n_pos, 1))
    w_neg = n / (2 * max(n_neg, 1))

    return w_pos, w_neg
end

function eval_mask_error_weighted(mask::Union{UInt64, UInt128},
                                 X::Matrix{UInt8},
                                 y::Vector{UInt8};
                                 w_pos::Float64=1.0,
                                 w_neg::Float64=1.0)

    n, m = size(X)

    if mask == 0
        n_pos = sum(y)
        n_neg = n - n_pos
        return min(w_pos*n_pos, w_neg*n_neg), 0
    end

    counts = zeros(Int16, n)
    k = 0

    @inbounds for j in 1:m
        if (mask >> (j-1)) & 1 == 1
            counts .+= X[:, j]
            k += 1
        end
    end

    err, tau = compute_error_weighted(counts, k, y;
                                    w_pos=w_pos,
                                    w_neg=w_neg)

    return err, tau
end

function gray_search_weighted(feats, y)

    n = length(y)
    m = length(feats)

    counts = zeros(Int16, n)

    w_pos, w_neg = compute_class_weights(y)

    best_err = Inf
    best_mask = UInt64(0)
    best_tau = 0

    prev_gray = UInt64(0)
    k = 0

    total = UInt64(1) << m

    best_err = min(w_pos*sum(y), w_neg*(n-sum(y)))

    @inbounds for t in UInt64(0):(total - 1)

        g = t ⊻ (t >> 1)

        if t > 0
            diff = g ⊻ prev_gray
            j = trailing_zeros(diff) + 1

            if ((g >> (j - 1)) & 1) == 1
                add_feature!(counts, feats[j])
                k += 1
            else
                remove_feature!(counts, feats[j])
                k -= 1
            end
        end

        if k == 0
            prev_gray = g
            continue
        end

        err, tau = compute_error_weighted(counts, k, y;
                                          w_pos=w_pos,
                                          w_neg=w_neg)

        if err < best_err ||
           (err == best_err && k < count_ones(best_mask))

            best_err = err
            best_mask = g
            best_tau = tau

            if best_err == 0
                break
            end
        end

        prev_gray = g
    end

    return best_mask, best_err, best_tau
end

function sffs_search_weighted(X::Matrix{UInt8}, y::Vector{UInt8};
                             max_k::Int=typemax(Int),
                             max_iter::Int=1000)

    w_pos, w_neg = compute_class_weights(y)

    function best_single()
        m = size(X, 2)
        best_j = 1
        best_err = Inf
        best_tau = 0

        for j in 1:m
            mask = UInt128(1) << (j-1)
            err, tau = eval_mask_error_weighted(mask, X, y;
                                           w_pos=w_pos,
                                           w_neg=w_neg)

            if err < best_err
                best_err = err
                best_j = j
                best_tau = tau
            end
        end

        return UInt128(1) << (best_j-1), best_err, best_tau
    end

    current_mask, current_err, current_tau = best_single()

    best_mask = current_mask
    best_err = current_err
    best_tau = current_tau

    iter = 0

    while iter < max_iter
        iter += 1
        improved = false

        # Forward
        best_add_err = current_err
        best_add_j = 0

        for j in 1:size(X, 2)
            if !has_bit(current_mask, j)
                new_mask = add_bit(current_mask, j)

                err, tau = eval_mask_error_weighted(new_mask, X, y;
                                               w_pos=w_pos,
                                               w_neg=w_neg)

                if err < best_add_err
                    best_add_err = err
                    best_add_j = j
                    best_tau = tau
                end
            end
        end

        if best_add_j != 0
            current_mask = add_bit(current_mask, best_add_j)
            current_err = best_add_err
            improved = true
        else
            break
        end

        # Backward
        while true
            best_remove_err = current_err
            best_remove_j = 0

            for j in 1:size(X, 2)
                if has_bit(current_mask, j)
                    new_mask = remove_bit(current_mask, j)

                    if new_mask == 0
                        continue
                    end

                    err, tau = eval_mask_error_weighted(new_mask, X, y;
                                                   w_pos=w_pos,
                                                   w_neg=w_neg)

                    if err < best_remove_err
                        best_remove_err = err
                        best_remove_j = j
                        best_tau = tau
                    end
                end
            end

            if best_remove_j != 0
                current_mask = remove_bit(current_mask, best_remove_j)
                current_err = best_remove_err
                improved = true
            else
                break
            end
        end

        if current_err < best_err ||
           (current_err == best_err &&
            count_ones(current_mask) < count_ones(best_mask))

            best_mask = current_mask
            best_err = current_err
        end

        if !improved || count_ones(current_mask) >= max_k
            break
        end
    end

    return best_mask, best_err, best_tau
end

function eval_mask_error_learnable(mask::Union{UInt64, UInt128}, X::Matrix{UInt8}, y::Vector{UInt8})

    n, m = size(X)

    if mask == 0
        n_pos = sum(y)
        n_neg = n - n_pos
        return min(n_pos, n_neg), 0
    end

    counts = zeros(Int16, n)
    k = 0

    @inbounds for j in 1:m
        if (mask >> (j-1)) & 1 == 1
            counts .+= X[:, j]
            k += 1
        end
    end

    err, tau = compute_error_learnable(counts, k, y)

    return err, tau
end

function best_single_feature(X, y)
    m = size(X, 2)

    best_j = 1
    best_err = typemax(Int)
    best_tau = 0

    for j in 1:m
        mask = UInt128(1) << (j-1)
        err, tau = eval_mask_error_learnable(mask, X, y)

        if err < best_err
            best_err = err
            best_j = j
            best_tau = tau
        end
    end

    return UInt128(1) << (best_j-1), best_err, best_tau
end

function sffs_search_learnable(X::Matrix{UInt8}, y::Vector{UInt8};
                               max_k::Int=typemax(Int),
                               max_iter::Int=1000)

    n, m = size(X)

    current_mask, current_err, current_tau = best_single_feature(X, y)

    best_mask = current_mask
    best_err = current_err
    best_tau = current_tau

    iter = 0

    while iter < max_iter
        iter += 1

        improved = false

        # Step 1: Forward inclusion
        best_add_err = current_err
        best_add_j = 0

        for j in 1:m
            if !has_bit(current_mask, j)

                new_mask = add_bit(current_mask, j)
                err, tau = eval_mask_error_learnable(new_mask, X, y)

                if err < best_add_err
                    best_add_err = err
                    best_add_j = j
                    best_tau = tau
                end
            end
        end

        if best_add_j != 0
            current_mask = add_bit(current_mask, best_add_j)
            current_err = best_add_err
            improved = true
        else
            break
        end

        # Step 2: Conditional backward
        while true
            best_remove_err = current_err
            best_remove_j = 0

            for j in 1:m
                if has_bit(current_mask, j)

                    new_mask = remove_bit(current_mask, j)

                    # Never return to empty set
                    if new_mask == 0
                        continue
                    end

                    err, tau = eval_mask_error_learnable(new_mask, X, y)

                    if err < best_remove_err
                        best_remove_err = err
                        best_remove_j = j
                        best_tau = tau
                    end
                end
            end

            if best_remove_j != 0
                current_mask = remove_bit(current_mask, best_remove_j)
                current_err = best_remove_err
                improved = true
            else
                break
            end
        end

        # Track global best
        if current_err < best_err ||
           (current_err == best_err &&
            count_ones(current_mask) < count_ones(best_mask))

            best_mask = current_mask
            best_err = current_err
        end

        if count_ones(current_mask) >= max_k
            break
        end

        if !improved
            break
        end
    end

    return best_mask, best_err, best_tau
end


# -------------------------
# Full pipeline
# -------------------------
"""
    select_feature_subset(X, y) -> (indices, error, p_value, tau)

Select the optimal subset of binary features via Gray code enumeration (≤20
features) or SFFS search (≤128 features), followed by a permutation test.
"""
function select_feature_subset(X::Matrix{UInt8}, y::Vector{UInt8})

    n, m = size(X)

    if m <= 20
        # Exact Gray code enumeration
        feats = build_features(X)
        feats = sort_features!(feats, y)
        mask, err, tau = gray_search_weighted(feats, y)
    elseif m <= 128
        # Sequential Floating Forward Selection
        mask, err, tau = sffs_search_weighted(X, y, max_k=15)
    else
        error("Too many features ($m > 128). Use RFMethod or LassoMethod instead.")
    end

    test = permutation_test_pvalue(mask, X, y, B=1000)

    return mask_to_indices(mask, m), err, test.p_value, tau
end
