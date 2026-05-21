"""Phase 2: corrected statistical methods for revenue A/B tests."""

from pathlib import Path
import math
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.simulate_users import simulate_users


FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def save_figure(fig, filename):
    """Save each figure in the shared folder used by the project dashboard."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_p_value(p_value):
    return f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"


def compare_ttest_vs_mannwhitney(df):
    """Compare mean-based and rank-based tests on revenue per user."""
    control = df.loc[df["group"] == "control", "post_experiment_revenue"]
    treatment = df.loc[df["group"] == "treatment", "post_experiment_revenue"]

    # Revenue per user is right-skewed in e-commerce. A few high-spending users can strongly
    # influence the mean, so a naive t-test can be sensitive to outliers. Mann-Whitney U provides a
    # non-parametric comparison that does not assume normally distributed revenue.
    ttest_result = stats.ttest_ind(control, treatment, equal_var=False)
    mannwhitney_result = stats.mannwhitneyu(
        control,
        treatment,
        alternative="two-sided",
    )

    return {
        "ttest_statistic": ttest_result.statistic,
        "ttest_pvalue": ttest_result.pvalue,
        "mannwhitney_statistic": mannwhitney_result.statistic,
        "mannwhitney_pvalue": mannwhitney_result.pvalue,
        "control_mean": control.mean(),
        "treatment_mean": treatment.mean(),
        "control_median": control.median(),
        "treatment_median": treatment.median(),
    }


def calculate_required_sample_size(effect_size, alpha=0.05, power=0.8):
    """Return required sample size per group for a two-arm experiment."""
    # Before running an experiment, DS teams estimate the required sample size so the test has enough
    # power to detect a meaningful business effect. Underpowered tests waste time and overpowered
    # tests can detect effects too small to matter.
    required_n = TTestIndPower().solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return math.ceil(required_n)


def cohens_d(control, treatment):
    """Calculate Cohen's d for treatment versus control."""
    control = np.asarray(control)
    treatment = np.asarray(treatment)

    control_variance = control.var(ddof=1)
    treatment_variance = treatment.var(ddof=1)
    control_n = len(control)
    treatment_n = len(treatment)

    # Pooled standard deviation puts the revenue lift on a business-comparable standardized scale.
    pooled_std = np.sqrt(
        ((control_n - 1) * control_variance + (treatment_n - 1) * treatment_variance)
        / (control_n + treatment_n - 2)
    )
    return (treatment.mean() - control.mean()) / pooled_std


def run_section_1(base_df):
    """Compare t-test and Mann-Whitney U on the main experiment dataset."""
    results = compare_ttest_vs_mannwhitney(base_df)

    # Statistical tests use the full revenue distribution; clipping is only for readable plotting
    # because revenue outliers can otherwise compress the business-relevant middle of the chart.
    plot_cutoff = base_df["post_experiment_revenue"].quantile(0.99)
    plot_df = base_df.assign(
        plotted_revenue=base_df["post_experiment_revenue"].clip(upper=plot_cutoff)
    )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot_df.boxplot(
        column="plotted_revenue",
        by="group",
        ax=ax,
        grid=False,
        showfliers=False,
    )
    ax.set_title("T-test vs. Mann-Whitney U on Skewed Revenue")
    ax.set_xlabel("Group")
    ax.set_ylabel("Post-experiment revenue, clipped at 99th percentile")
    ax.text(
        0.5,
        0.95,
        (
            f"t-test p = {format_p_value(results['ttest_pvalue'])}\n"
            f"Mann-Whitney U p = {format_p_value(results['mannwhitney_pvalue'])}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    fig.suptitle("")
    fig.tight_layout()
    save_figure(fig, "m2_ttest_vs_mannwhitney.png")

    print("\nSECTION 1 — T-TEST VS MANN-WHITNEY U")
    print(f"Control mean revenue: {results['control_mean']:.2f}")
    print(f"Treatment mean revenue: {results['treatment_mean']:.2f}")
    print(f"Control median revenue: {results['control_median']:.2f}")
    print(f"Treatment median revenue: {results['treatment_median']:.2f}")
    print(f"T-test p-value: {format_p_value(results['ttest_pvalue'])}")
    print(f"Mann-Whitney U p-value: {format_p_value(results['mannwhitney_pvalue'])}")
    print(
        "Correct interpretation: Mann-Whitney U is more robust for skewed revenue because it "
        "compares group distributions/ranks without assuming normal revenue."
    )


def run_section_2():
    """Demonstrate power analysis across plausible standardized effects."""
    effect_sizes = [0.01, 0.02, 0.05, 0.10, 0.20]
    required_samples = {
        effect_size: calculate_required_sample_size(effect_size)
        for effect_size in effect_sizes
    }

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        list(required_samples.keys()),
        list(required_samples.values()),
        marker="o",
    )
    ax.set_title("Required Sample Size Increases as Effect Size Gets Smaller")
    ax.set_xlabel("Cohen's d effect size")
    ax.set_ylabel("Required sample size per group")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "m2_power_analysis.png")

    print("\nSECTION 2 — POWER ANALYSIS")
    print("Required sample size per group:")
    for effect_size, sample_size in required_samples.items():
        print(f"d = {effect_size:.2f}: {sample_size}")
    print(
        "Correct interpretation: Smaller effects require much larger samples. This is why "
        "production A/B tests need power analysis before launch."
    )


def run_section_3():
    """Contrast statistical significance with practical business significance."""
    # With very large sample sizes, tiny effects can produce p < 0.05 even when the business impact is
    # negligible. A good DS does not stop at statistical significance; they also evaluate practical
    # significance.
    sample_size = 100000
    true_effect_size = 0.005
    large_df = simulate_users(
        n_users=sample_size,
        seed=123,
        effect_size=true_effect_size,
    )
    control = large_df.loc[large_df["group"] == "control", "post_experiment_revenue"]
    treatment = large_df.loc[large_df["group"] == "treatment", "post_experiment_revenue"]

    ttest_result = stats.ttest_ind(control, treatment, equal_var=False)
    mannwhitney_result = stats.mannwhitneyu(
        control,
        treatment,
        alternative="two-sided",
    )
    effect_d = cohens_d(control, treatment)
    control_mean = control.mean()
    treatment_mean = treatment.mean()
    revenue_lift = (treatment_mean / control_mean - 1) * 100

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(["Control", "Treatment"], [control_mean, treatment_mean])
    ax.set_title("Statistical Significance Does Not Always Mean Practical Significance")
    ax.set_ylabel("Mean post-experiment revenue")
    ax.text(
        0.04,
        0.82,
        (
            f"t-test p = {format_p_value(ttest_result.pvalue)}\n"
            f"Cohen's d = {effect_d:.4f}\n"
            f"Observed revenue lift = {revenue_lift:.2f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    save_figure(fig, "m2_statistical_vs_practical.png")

    print("\nSECTION 3 — STATISTICAL VS PRACTICAL SIGNIFICANCE")
    print(f"Sample size: {sample_size} users")
    print(f"True simulated lift: {true_effect_size * 100:.1f}%")
    print(f"T-test p-value: {format_p_value(ttest_result.pvalue)}")
    print(f"Mann-Whitney U p-value: {format_p_value(mannwhitney_result.pvalue)}")
    print(f"Cohen's d: {effect_d:.4f}")
    print(
        "Naive conclusion: The analyst would call this statistically significant because p < 0.05."
    )
    print(
        "Correct conclusion: The effect is practically negligible because Cohen's d is extremely "
        "small and the revenue lift is tiny."
    )
    print(
        "Why they differ: Large sample sizes can make very small differences statistically "
        "significant, but DS teams must decide whether the effect is large enough to matter for the "
        "business."
    )


def main():
    # The fixed business scenario keeps Phase 2 directly comparable to Phase 1's naive analyses.
    base_df = simulate_users(n_users=10000, seed=42, effect_size=0.05)

    run_section_1(base_df)
    run_section_2()
    run_section_3()


if __name__ == "__main__":
    main()
