"""Bayesian reasoning + parameter estimation (syllabus: Bayes rule, MLE)."""
from scipy.stats import beta


def mle_rate(successes, n):
    """Maximum-likelihood estimate of a merge rate: s/n. Wild for small n (2 of 2 => 100%)."""
    return successes / n if n else float("nan")


def posterior(successes, n, prior_rate, strength=10):
    """Beta-Binomial MAP/mean. Prior = global merge rate worth `strength` pseudo-observations."""
    a0, b0 = prior_rate * strength, (1 - prior_rate) * strength
    a, b = a0 + successes, b0 + (n - successes)
    return a / (a + b), tuple(beta.interval(0.9, a, b))


def update(prior, *likelihood_ratios):
    """Bayes rule in odds form: posterior odds = prior odds x LR1 x LR2 ..."""
    odds = prior / (1 - prior)
    for lr in likelihood_ratios:
        odds *= lr
    return odds / (1 + odds)


def demo_table(strength=10):
    """Per-repo MLE vs Bayesian estimate, from your own database (shows shrinkage for small repos)."""
    import pandas as pd
    from collector.db import conn
    d = pd.read_sql("SELECT repo, pr_merged FROM issues WHERE pr_merged IS NOT NULL", conn())
    g = d.groupby("repo").pr_merged.agg(["sum", "count"]).rename(columns={"sum": "merged", "count": "n"})
    prior = g.merged.sum() / g.n.sum()
    g["mle"] = [mle_rate(m, n) for m, n in zip(g.merged, g.n)]
    post = [posterior(m, n, prior, strength) for m, n in zip(g.merged, g.n)]
    g["bayes"] = [p[0] for p in post]; g["ci90_low"] = [p[1][0] for p in post]; g["ci90_high"] = [p[1][1] for p in post]
    return prior, g.sort_values("n").round(3)