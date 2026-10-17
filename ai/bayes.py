from scipy.stats import beta


def posterior(successes, n, prior_rate, strength=10):
    """Beta-Binomial. Prior = global merge rate worth `strength` pseudo-observations."""
    a0, b0 = prior_rate * strength, (1 - prior_rate) * strength
    a, b = a0 + successes, b0 + (n - successes)
    return a / (a + b), tuple(beta.interval(0.9, a, b))


def update(prior, *likelihood_ratios):
    """Bayes rule in odds form: posterior odds = prior odds x LR1 x LR2 ..."""
    odds = prior / (1 - prior)
    for lr in likelihood_ratios:
        odds *= lr
    return odds / (1 + odds)