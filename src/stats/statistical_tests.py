"""Statistical hypothesis testing toolkit for cohort comparisons.

Every function returns real computed statistics (scipy/statsmodels) plus a
plain-English interpretation string suitable for a non-technical business
audience. Nothing here is hardcoded — inputs are numpy arrays / pandas
Series pulled live from the SQLite database or a filtered cohort.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    plain_english: str
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "test_name": self.test_name,
            "statistic": round(float(self.statistic), 4),
            "p_value": round(float(self.p_value), 6),
            "alpha": self.alpha,
            "significant": self.significant,
            "plain_english": self.plain_english,
            **{k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in self.extra.items()},
        }


def welch_t_test(
    group_a: np.ndarray, group_b: np.ndarray,
    label_a: str = "Drug X", label_b: str = "Placebo",
    metric_name: str = "primary outcome score", alpha: float = 0.05,
) -> TestResult:
    """Welch's t-test (unequal variances) comparing two independent groups."""
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    mean_a, mean_b = group_a.mean(), group_b.mean()
    d = cohens_d(group_a, group_b)
    sig = bool(p_value < alpha)

    direction = "higher" if mean_a > mean_b else "lower"
    chance_pct = "0.1%" if p_value < 0.001 else f"{p_value*100:.1f}%"
    plain = (
        f"On average, the {label_a} arm scored {abs(mean_a - mean_b):.2f} points {direction} than "
        f"{label_b} on {metric_name} ({mean_a:.2f} vs {mean_b:.2f}). "
        f"{'This difference is statistically significant' if sig else 'This difference is not statistically significant'} "
        f"(Welch's t={t_stat:.2f}, p={_fmt_p(p_value)}), meaning "
        f"{'we would see a difference this large by chance in fewer than ' + chance_pct + ' of repeated trials' if sig else 'random variation alone could plausibly explain the observed difference'}. "
        f"The effect size is {d.extra['interpretation']} (Cohen's d = {d.statistic:.2f})."
    )

    return TestResult(
        test_name="Welch's t-test",
        statistic=t_stat,
        p_value=p_value,
        alpha=alpha,
        significant=sig,
        plain_english=plain,
        extra={
            f"mean_{label_a.lower().replace(' ', '_')}": mean_a,
            f"mean_{label_b.lower().replace(' ', '_')}": mean_b,
            "cohens_d": d.statistic,
            "effect_size_interpretation": d.extra["interpretation"],
            "n_a": len(group_a),
            "n_b": len(group_b),
        },
    )


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> TestResult:
    """Cohen's d effect size for two independent samples (pooled SD)."""
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    n_a, n_b = len(group_a), len(group_b)
    pooled_sd = np.sqrt(
        ((n_a - 1) * group_a.var(ddof=1) + (n_b - 1) * group_b.var(ddof=1)) / (n_a + n_b - 2)
    )
    d = (group_a.mean() - group_b.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    plain = f"Cohen's d = {d:.2f}, a {interp} effect size."
    return TestResult(
        test_name="Cohen's d", statistic=d, p_value=float("nan"), alpha=0.0,
        significant=abs_d >= 0.2, plain_english=plain, extra={"interpretation": interp},
    )


def chi_square_test(
    group_labels: np.ndarray, category_labels: np.ndarray,
    metric_name: str = "adverse event rate", alpha: float = 0.05,
) -> TestResult:
    """Chi-square test of independence between arm and a categorical outcome
    (e.g. AE occurred / did not occur, or AE severity class)."""
    import pandas as pd

    contingency = pd.crosstab(group_labels, category_labels)
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    sig = bool(p_value < alpha)

    rates = pd.crosstab(group_labels, category_labels, normalize="index")

    plain = (
        f"A chi-square test of independence was run on {metric_name} across treatment arms "
        f"(chi2={chi2:.2f}, dof={dof}, p={_fmt_p(p_value)}). "
        f"{'There is a statistically significant association' if sig else 'There is no statistically significant association'} "
        f"between treatment arm and {metric_name}."
    )

    return TestResult(
        test_name="Chi-square test of independence",
        statistic=chi2, p_value=p_value, alpha=alpha, significant=sig, plain_english=plain,
        extra={"degrees_of_freedom": dof, "contingency_table": contingency.to_dict(), "rates_by_arm": rates.to_dict()},
    )


def kruskal_wallis_test(
    *groups: np.ndarray, group_names: Optional[list] = None,
    metric_name: str = "lab value", alpha: float = 0.05,
) -> TestResult:
    """Kruskal-Wallis H-test: non-parametric alternative to one-way ANOVA,
    used when the normality assumption for a t-test/ANOVA does not hold."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    h_stat, p_value = stats.kruskal(*groups)
    sig = bool(p_value < alpha)

    names = group_names or [f"Group {i+1}" for i in range(len(groups))]
    medians = {name: float(np.median(g)) for name, g in zip(names, groups)}

    plain = (
        f"A Kruskal-Wallis test compared {metric_name} across {len(groups)} groups without assuming "
        f"normally distributed data (H={h_stat:.2f}, p={_fmt_p(p_value)}). "
        f"{'The groups differ significantly' if sig else 'The groups do not differ significantly'} "
        f"in {metric_name}. Median values: " + ", ".join(f"{k}={v:.2f}" for k, v in medians.items()) + "."
    )

    return TestResult(
        test_name="Kruskal-Wallis H-test", statistic=h_stat, p_value=p_value, alpha=alpha,
        significant=sig, plain_english=plain, extra={"medians": medians},
    )


def normality_check(data: np.ndarray, metric_name: str = "metric", alpha: float = 0.05) -> TestResult:
    """Shapiro-Wilk normality test — used to decide whether a t-test/ANOVA or
    a non-parametric Kruskal-Wallis/Mann-Whitney test is appropriate."""
    data = np.asarray(data, dtype=float)
    sample = data if len(data) <= 5000 else np.random.default_rng(0).choice(data, 5000, replace=False)
    stat, p_value = stats.shapiro(sample)
    normal = bool(p_value >= alpha)

    plain = (
        f"Shapiro-Wilk test on {metric_name} (W={stat:.3f}, p={_fmt_p(p_value)}): "
        f"data {'appears normally distributed' if normal else 'deviates significantly from a normal distribution'} "
        f"— {'a t-test/ANOVA is appropriate' if normal else 'a non-parametric test (Kruskal-Wallis / Mann-Whitney) is recommended'}."
    )
    return TestResult(
        test_name="Shapiro-Wilk normality test", statistic=stat, p_value=p_value, alpha=alpha,
        significant=not normal, plain_english=plain, extra={"is_normal": normal},
    )


def confidence_interval(data: np.ndarray, confidence: float = 0.95, metric_name: str = "metric") -> dict:
    """Mean and (1 - alpha) confidence interval via the t-distribution."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    mean = data.mean()
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    lower, upper = mean - margin, mean + margin

    plain = (
        f"The {int(confidence*100)}% confidence interval for {metric_name} is "
        f"[{lower:.2f}, {upper:.2f}] around a sample mean of {mean:.2f} (n={n})."
    )
    return {
        "metric_name": metric_name, "mean": round(float(mean), 4), "lower": round(float(lower), 4),
        "upper": round(float(upper), 4), "confidence": confidence, "n": n, "plain_english": plain,
    }
