"""Phase 1: naive A/B testing failures for revenue-per-user experiments."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.simulate_users import simulate_users


FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def save_figure(fig, filename):
    """Save figures to the shared output folder used by later project phases."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _welch_ttest_p_value(control, treatment):
    # Welch's t-test is the common naive default because experiment groups may have unequal variance.
    return stats.ttest_ind(control, treatment, equal_var=False).pvalue


def simulate_daily_peeking(seed=42, effect_size=0.05, n_users=10000, n_days=14):
    """Simulate daily p-values when an analyst repeatedly checks an experiment."""
    df = simulate_users(n_users=n_users, seed=seed, effect_size=effect_size).copy()
    rng = np.random.default_rng(seed + 10_000)

    # Users arrive across calendar days, so peeking mimics a dashboard checked before full enrollment.
    df["experiment_day"] = rng.integers(1, n_days + 1, size=n_users)

    results = []
    for day in range(1, n_days + 1):
        cumulative_df = df[df["experiment_day"] <= day]
        control = cumulative_df.loc[
            cumulative_df["group"] == "control", "post_experiment_revenue"
        ]
        treatment = cumulative_df.loc[
            cumulative_df["group"] == "treatment", "post_experiment_revenue"
        ]

        p_value = _welch_ttest_p_value(control, treatment)
        results.append(
            {
                "day": day,
                "cumulative_users": len(cumulative_df),
                "p_value": p_value,
            }
        )

    return pd.DataFrame(results)


def estimate_false_positive_rate_under_peeking(
    n_experiments=1000, n_users=10000, n_days=14, alpha=0.05
):
    """Estimate how often repeated daily checks create false positives."""
    false_positives = 0

    for experiment_id in range(n_experiments):
        # A zero true lift isolates Type I error: any significant result is random noise.
        peeking_results = simulate_daily_peeking(
            seed=42 + experiment_id,
            effect_size=0.0,
            n_users=n_users,
            n_days=n_days,
        )
        if (peeking_results["p_value"] < alpha).any():
            false_positives += 1

    return false_positives / n_experiments


def create_device_level_contamination(df):
    """Duplicate multi-device users into the opposite group to mimic device randomization."""
    # Multi-device shoppers can be split across variants if assignment happens per device, not per person.
    multi_device_users = df[df["device_count"] >= 2]
    duplicated_users = multi_device_users.copy()
    duplicated_users["group"] = np.where(
        duplicated_users["group"] == "control", "treatment", "control"
    )

    contaminated_df = pd.concat([df, duplicated_users], ignore_index=True)
    return contaminated_df, len(multi_device_users)


def _format_p_value(p_value):
    return f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"


def _significance_label(p_value, alpha=0.05):
    return "significant" if p_value < alpha else "not significant"


def run_failure_1(n_experiments_for_main=300):
    """Show how repeated peeking inflates the false positive rate."""
    # Analysts often check experiment results before the planned end date and stop as soon as p < 0.05.
    # This inflates the false positive rate because each repeated check creates another chance to find
    # significance by random noise.
    peeking_results = simulate_daily_peeking(seed=42, effect_size=0.05)

    print(
        f"NOTE: Estimating peeking false positive rate with {n_experiments_for_main} "
        "experiments in __main__ for runtime; function default remains 1000."
    )
    false_positive_rate = estimate_false_positive_rate_under_peeking(
        n_experiments=n_experiments_for_main
    )

    day_3_p = peeking_results.loc[peeking_results["day"] == 3, "p_value"].iloc[0]
    day_7_p = peeking_results.loc[peeking_results["day"] == 7, "p_value"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(peeking_results["day"], peeking_results["p_value"], marker="o")
    axes[0].axhline(0.05, linestyle="--", color="black", linewidth=1)
    axes[0].annotate(
        f"Day 3: {_format_p_value(day_3_p)}",
        xy=(3, day_3_p),
        xytext=(3.5, min(day_3_p + 0.08, 0.95)),
        arrowprops={"arrowstyle": "->", "linewidth": 1},
    )
    axes[0].annotate(
        f"Day 7: {_format_p_value(day_7_p)}",
        xy=(7, day_7_p),
        xytext=(7.5, min(day_7_p + 0.08, 0.95)),
        arrowprops={"arrowstyle": "->", "linewidth": 1},
    )
    axes[0].set_title("P-value Over Time: Daily Peeking")
    axes[0].set_xlabel("Experiment day")
    axes[0].set_ylabel("p-value")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(range(1, 15))

    axes[1].bar(
        ["Naive peeking\nfalse positive rate", "Correct alpha\n= 5%"],
        [false_positive_rate * 100, 5.0],
    )
    axes[1].set_title("False Positive Rate: Naive Peeking vs. Correct Alpha")
    axes[1].set_ylabel("False positive rate (%)")
    axes[1].set_ylim(0, max(false_positive_rate * 100, 5.0) * 1.35)

    fig.tight_layout()
    save_figure(fig, "m1_peeking.png")

    print("\nFAILURE 1 — PEEKING")
    print(
        f"Naive conclusion: Day 3 p-value = {_format_p_value(day_3_p)}, "
        f"Day 7 p-value = {_format_p_value(day_7_p)}"
    )
    print(f"True false positive rate under peeking: {false_positive_rate * 100:.1f}%")
    print("Correct false positive rate without repeated peeking: 5.0%")
    print(
        "Why they differ: Repeated testing inflates Type I error because each "
        "additional look is another chance to cross the significance threshold by random noise."
    )

    return false_positive_rate


def run_failure_2(base_df):
    """Show how device-level assignment contaminates user-level experiments."""
    # Users who own multiple devices can be assigned inconsistently if randomization happens at the
    # device level instead of the user level. This contaminates the experiment because the same real
    # person may appear in both control and treatment.
    contaminated_df, affected_users = create_device_level_contamination(base_df)

    clean_control = base_df.loc[base_df["group"] == "control", "post_experiment_revenue"]
    clean_treatment = base_df.loc[
        base_df["group"] == "treatment", "post_experiment_revenue"
    ]
    contaminated_control = contaminated_df.loc[
        contaminated_df["group"] == "control", "post_experiment_revenue"
    ]
    contaminated_treatment = contaminated_df.loc[
        contaminated_df["group"] == "treatment", "post_experiment_revenue"
    ]

    clean_p = _welch_ttest_p_value(clean_control, clean_treatment)
    contaminated_p = _welch_ttest_p_value(contaminated_control, contaminated_treatment)

    # The statistical tests use full revenue data; clipping below is only for readable box plots.
    plot_cutoff = contaminated_df["post_experiment_revenue"].quantile(0.99)
    clean_plot = base_df.assign(
        plotted_revenue=base_df["post_experiment_revenue"].clip(upper=plot_cutoff)
    )
    contaminated_plot = contaminated_df.assign(
        plotted_revenue=contaminated_df["post_experiment_revenue"].clip(
            upper=plot_cutoff
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, plot_df, title, p_value in [
        (axes[0], clean_plot, "Clean user-level randomization", clean_p),
        (
            axes[1],
            contaminated_plot,
            "Contaminated device-level randomization",
            contaminated_p,
        ),
    ]:
        plot_df.boxplot(
            column="plotted_revenue",
            by="group",
            ax=ax,
            grid=False,
            showfliers=False,
        )
        ax.set_title(title)
        ax.set_xlabel("Group")
        ax.set_ylabel("Post-experiment revenue")
        ax.text(
            0.5,
            0.95,
            f"p = {_format_p_value(p_value)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

    fig.suptitle("Impact of Imperfect Randomization on Revenue Distributions")
    fig.tight_layout()
    save_figure(fig, "m1_randomization.png")

    print("\nFAILURE 2 — IMPERFECT RANDOMIZATION")
    print(f"Clean user-level p-value: {_format_p_value(clean_p)}")
    print(f"Contaminated device-level p-value: {_format_p_value(contaminated_p)}")
    print(f"Number of multi-device users affected: {affected_users}")
    print(f"Naive conclusion: {_significance_label(contaminated_p)}")
    print(f"Correct conclusion: {_significance_label(clean_p)}")
    print(
        "Why they differ: Device-level assignment can place the same real user in both groups, "
        "contaminating the treatment comparison. The correction is to randomize at the user level."
    )


def run_failure_3(base_df):
    """Show why skewed revenue makes naive t-test conclusions less robust."""
    # Revenue per user in e-commerce is heavily right-skewed. Most users spend little, while a small
    # number of users spend much more. This makes naive average-based testing sensitive to outliers
    # and motivates more robust methods.
    control = base_df.loc[base_df["group"] == "control", "post_experiment_revenue"]
    treatment = base_df.loc[base_df["group"] == "treatment", "post_experiment_revenue"]
    revenue = base_df["post_experiment_revenue"]
    log_revenue = np.log1p(revenue)

    raw_skewness = stats.skew(revenue)
    log_skewness = stats.skew(log_revenue)
    ttest_p = _welch_ttest_p_value(control, treatment)

    rng = np.random.default_rng(42)
    sample_size = min(5000, len(control))
    sampled_control = rng.choice(control.to_numpy(), size=sample_size, replace=False)
    # Shapiro-Wilk is interpreted with skewness, histograms, and Q-Q plots because large samples can
    # flag tiny deviations from normality that may not be the main business risk.
    shapiro_p = stats.shapiro(sampled_control).pvalue
    normality_reasonable = raw_skewness < 1 and shapiro_p >= 0.05

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    bins = 60
    axes[0].hist(control, bins=bins, density=True, alpha=0.55, label="Control")
    axes[0].hist(treatment, bins=bins, density=True, alpha=0.45, label="Treatment")
    x_values = np.linspace(control.min(), control.quantile(0.995), 300)
    normal_curve = stats.norm.pdf(x_values, loc=control.mean(), scale=control.std())
    axes[0].plot(x_values, normal_curve, color="black", linewidth=1.5, label="Control normal fit")
    axes[0].set_xlim(0, revenue.quantile(0.995))
    axes[0].set_title("Revenue Distribution — Raw")
    axes[0].set_xlabel("Post-experiment revenue")
    axes[0].set_ylabel("Density")
    axes[0].annotate(f"Raw skewness: {raw_skewness:.2f}", xy=(0.05, 0.9), xycoords="axes fraction")
    axes[0].legend()

    axes[1].hist(np.log1p(control), bins=bins, density=True, alpha=0.55, label="Control")
    axes[1].hist(
        np.log1p(treatment), bins=bins, density=True, alpha=0.45, label="Treatment"
    )
    axes[1].set_title("Revenue Distribution — Log Transformed")
    axes[1].set_xlabel("log1p(post-experiment revenue)")
    axes[1].set_ylabel("Density")
    axes[1].annotate(
        f"Log skewness: {log_skewness:.2f}", xy=(0.05, 0.9), xycoords="axes fraction"
    )
    axes[1].legend()

    stats.probplot(control, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot: Control Revenue vs. Normal")
    axes[2].get_lines()[1].set_color("black")

    fig.tight_layout()
    save_figure(fig, "m1_skewness.png")

    print("\nFAILURE 3 — REVENUE SKEWNESS")
    print(f"Skewness of raw revenue distribution: {raw_skewness:.2f}")
    print(f"Skewness after log transform: {log_skewness:.2f}")
    print(f"Shapiro-Wilk p-value on sampled control revenue: {_format_p_value(shapiro_p)}")
    print(f"Normality assumption appears reasonable: {'Yes' if normality_reasonable else 'No'}")
    print(f"T-test p-value: {_format_p_value(ttest_p)}")
    print(
        "Naive conclusion: "
        f"The treatment effect is {_significance_label(ttest_p)} based on the t-test."
    )
    print(
        "Correct conclusion: Revenue is heavily right-skewed, so naive average-based testing can "
        "be sensitive to outliers. Module 2 will compare this with Mann-Whitney U as a more robust "
        "non-parametric method."
    )
    print(
        "Why they differ: The histogram, skewness, and Q-Q plot show that revenue does not follow "
        "a normal distribution, so relying only on a naive t-test can be misleading."
    )


def main():
    # The fixed scenario keeps all three demonstrations comparable: same e-commerce experiment,
    # same revenue-per-user metric, same seed, and same 5% recommendation lift.
    base_df = simulate_users(n_users=10000, seed=42, effect_size=0.05)

    false_positive_rate = run_failure_1()
    if false_positive_rate <= 0.05:
        print(
            "WARNING: Estimated peeking false positive rate was not above 5%; "
            "increase n_experiments_for_main for a more stable estimate."
        )

    run_failure_2(base_df)
    run_failure_3(base_df)


if __name__ == "__main__":
    main()
