"""Phase 4: planned sequential testing for revenue A/B experiments."""

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
    """Save figures to the shared project output folder."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_p_value(p_value):
    return f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"


def alpha_spending_threshold(day, n_days, total_alpha=0.05):
    """Return the alpha allocated to a planned interim look."""
    # Product teams often want to stop experiments early when results are clearly positive or clearly
    # negative. Sequential testing makes this possible only if the stopping rule is planned in
    # advance. Otherwise, repeated peeking inflates the false positive rate.
    # This educational implementation uses incremental alpha spending. Each look receives only the
    # alpha allocated to that look, which is more conservative than comparing every interim p-value
    # to 0.05 and helps control false positives.
    cumulative_alpha_t = total_alpha * (day / n_days) ** 2
    cumulative_alpha_prev = total_alpha * ((day - 1) / n_days) ** 2
    return cumulative_alpha_t - cumulative_alpha_prev


def _assign_experiment_days(df, seed, n_days):
    rng = np.random.default_rng(seed + 40_000)
    assigned_df = df.copy()
    # Daily arrivals mimic a running production experiment dashboard with cumulative user exposure.
    assigned_df["experiment_day"] = rng.integers(1, n_days + 1, size=len(df))
    return assigned_df


def _p_value_by_group(df, metric):
    control = df.loc[df["group"] == "control", metric]
    treatment = df.loc[df["group"] == "treatment", metric]
    # Welch's t-test is used because online revenue variance can differ between experiment arms.
    return stats.ttest_ind(control, treatment, equal_var=False).pvalue


def run_sequential_test(
    seed=42,
    n_users=10000,
    effect_size=0.03,
    n_days=14,
    total_alpha=0.05,
):
    """Run an alpha-spending sequential test for any treatment difference."""
    df = simulate_users(n_users=n_users, seed=seed, effect_size=effect_size)
    df = _assign_experiment_days(df, seed, n_days)

    stop_day = None
    stop_reason = "continue"
    rows = []

    for day in range(1, n_days + 1):
        cumulative_df = df[df["experiment_day"] <= day]
        p_value = _p_value_by_group(cumulative_df, "post_experiment_revenue")
        alpha_threshold = alpha_spending_threshold(day, n_days, total_alpha)

        decision = "continue"
        if stop_day is None and p_value < alpha_threshold:
            decision = "stop_for_significance"
            stop_day = day
            stop_reason = decision
        elif stop_day is not None:
            decision = stop_reason

        rows.append(
            {
                "day": day,
                "cumulative_users": len(cumulative_df),
                "p_value": p_value,
                "alpha_threshold": alpha_threshold,
                "decision": decision,
            }
        )

    return pd.DataFrame(rows), stop_day, stop_reason


def run_sequential_harm_test(
    seed=42,
    n_users=10000,
    effect_size=-0.05,
    n_days=14,
    total_alpha=0.05,
):
    """Run an alpha-spending sequential test that stops only for treatment harm."""
    df = simulate_users(n_users=n_users, seed=seed, effect_size=effect_size)
    df = _assign_experiment_days(df, seed, n_days)

    stop_day = None
    stop_reason = "continue"
    rows = []

    for day in range(1, n_days + 1):
        cumulative_df = df[df["experiment_day"] <= day]
        control = cumulative_df.loc[
            cumulative_df["group"] == "control", "post_experiment_revenue"
        ]
        treatment = cumulative_df.loc[
            cumulative_df["group"] == "treatment", "post_experiment_revenue"
        ]
        control_mean = control.mean()
        treatment_mean = treatment.mean()
        # Harm monitoring protects users only when the treatment is both directionally worse and
        # statistically beyond the pre-planned alpha-spending boundary.
        p_value = stats.ttest_ind(control, treatment, equal_var=False).pvalue
        alpha_threshold = alpha_spending_threshold(day, n_days, total_alpha)

        decision = "continue"
        if (
            stop_day is None
            and treatment_mean < control_mean
            and p_value < alpha_threshold
        ):
            decision = "stop_for_harm"
            stop_day = day
            stop_reason = decision
        elif stop_day is not None:
            decision = stop_reason

        rows.append(
            {
                "day": day,
                "cumulative_users": len(cumulative_df),
                "p_value": p_value,
                "alpha_threshold": alpha_threshold,
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "decision": decision,
            }
        )

    return pd.DataFrame(rows), stop_day, stop_reason


def find_winning_demo_case():
    """Find a reproducible early-stop case where treatment is clearly winning."""
    # If a treatment is clearly improving revenue, stopping early lets the company ship the better
    # recommendation algorithm sooner.
    best_case = None
    for seed in range(1, 201):
        for effect_size in [0.06, 0.08, 0.10, 0.12]:
            results_df, stop_day, stop_reason = run_sequential_test(
                seed=seed,
                effect_size=effect_size,
            )
            if stop_day is not None and stop_day < 14:
                stop_row = results_df.loc[results_df["day"] == stop_day].iloc[0]
                if stop_row["p_value"] < stop_row["alpha_threshold"]:
                    candidate = (seed, effect_size, results_df, stop_day, stop_reason)
                    if best_case is None or stop_day < best_case[3]:
                        best_case = candidate

    if best_case is not None:
        return best_case

    raise RuntimeError("No winning sequential testing demo case found.")


def find_losing_demo_case():
    """Find a reproducible early-stop case where treatment is clearly losing."""
    # If a treatment is clearly hurting revenue, stopping early reduces how long users are exposed to
    # a worse product experience.
    best_case = None
    for seed in range(1, 201):
        for effect_size in [-0.06, -0.08, -0.10, -0.12]:
            results_df, stop_day, stop_reason = run_sequential_harm_test(
                seed=seed,
                effect_size=effect_size,
            )
            if stop_day is not None and stop_day < 14:
                candidate = (seed, effect_size, results_df, stop_day, stop_reason)
                if best_case is None or stop_day < best_case[3]:
                    best_case = candidate

    if best_case is not None:
        return best_case

    raise RuntimeError("No losing sequential testing demo case found.")


def compare_false_positive_rates(
    n_experiments=500,
    n_users=10000,
    n_days=14,
    alpha=0.05,
):
    """Compare false positives from naive peeking and planned alpha spending."""
    naive_false_positives = 0
    sequential_false_positives = 0

    for experiment_id in range(n_experiments):
        # The main danger of naive peeking is that it turns random noise into false discoveries.
        # A sequential correction should keep the false positive rate closer to the intended alpha.
        seed = 10_000 + experiment_id
        df = simulate_users(n_users=n_users, seed=seed, effect_size=0.0)
        df = _assign_experiment_days(df, seed, n_days)

        naive_hit = False
        sequential_hit = False
        for day in range(1, n_days + 1):
            cumulative_df = df[df["experiment_day"] <= day]
            p_value = _p_value_by_group(cumulative_df, "post_experiment_revenue")

            if p_value < alpha:
                naive_hit = True

            if p_value < alpha_spending_threshold(day, n_days, alpha):
                sequential_hit = True

        naive_false_positives += int(naive_hit)
        sequential_false_positives += int(sequential_hit)

    return {
        "naive_false_positive_rate": naive_false_positives / n_experiments,
        "sequential_false_positive_rate": sequential_false_positives / n_experiments,
        "alpha": alpha,
    }


def _plot_sequential_results(results_df, stop_day, filename, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["day"], results_df["p_value"], marker="o", label="Observed p-value")
    ax.plot(
        results_df["day"],
        results_df["alpha_threshold"],
        marker="o",
        label="Per-look alpha-spending threshold",
    )
    ax.axhline(0.05, linestyle="--", color="black", linewidth=1, label="p = 0.05")
    ax.set_title(title)
    ax.set_xlabel("Experiment day")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, min(1, max(results_df["p_value"].max(), 0.08) * 1.25))
    ax.set_xticks(results_df["day"])
    ax.legend()

    if stop_day is not None:
        stop_row = results_df.loc[results_df["day"] == stop_day].iloc[0]
        ax.annotate(
            f"Stop: day {stop_day}",
            xy=(stop_day, stop_row["p_value"]),
            xytext=(stop_day + 0.7, min(stop_row["p_value"] + 0.08, ax.get_ylim()[1] * 0.9)),
            arrowprops={"arrowstyle": "->", "linewidth": 1},
        )

    fig.tight_layout()
    save_figure(fig, filename)


def run_section_1():
    """Print a basic sequential testing implementation summary."""
    results_df, stop_day, stop_reason = run_sequential_test()
    final_row = results_df.iloc[-1]

    print("\nSECTION 1 — SEQUENTIAL TEST IMPLEMENTATION")
    print(f"First stopping day: {stop_day if stop_day is not None else 'Not stopped'}")
    print(f"Stopping reason: {stop_reason}")
    print(f"Final-day p-value: {format_p_value(final_row['p_value'])}")
    print(f"Final per-look alpha-spending threshold: {final_row['alpha_threshold']:.4f}")
    print(
        "Correct interpretation: This educational alpha-spending correction demonstrates the core "
        "idea that early looks use stricter thresholds than naive p < 0.05."
    )


def run_section_2():
    """Demonstrate planned early stopping for a winning treatment."""
    seed, effect_size, results_df, stop_day, stop_reason = find_winning_demo_case()
    _plot_sequential_results(
        results_df,
        stop_day,
        "m4_winning_early_stop.png",
        "Sequential Testing: Early Stop When Treatment Is Winning",
    )

    print("\nSECTION 2 — EARLY STOPPING: TREATMENT WINNING")
    print(f"Seed used: {seed}")
    print(f"Effect size used: {effect_size:.2f}")
    print(f"Stopping day: {stop_day}")
    print(f"Stopping reason: {stop_reason}")
    print("Naive interpretation: The treatment appears beneficial.")
    print(
        "Correct interpretation: Because the stopping rule was planned and each look receives only "
        "its allocated alpha, early stopping is statistically defensible in this educational setup."
    )


def run_section_3():
    """Demonstrate planned early stopping for a harmful treatment."""
    seed, effect_size, results_df, stop_day, stop_reason = find_losing_demo_case()
    _plot_sequential_results(
        results_df,
        stop_day,
        "m4_losing_early_stop.png",
        "Sequential Testing: Early Stop When Treatment Is Losing",
    )

    print("\nSECTION 3 — EARLY STOPPING: TREATMENT LOSING")
    print(f"Seed used: {seed}")
    print(f"Effect size used: {effect_size:.2f}")
    print(f"Stopping day: {stop_day}")
    print(f"Stopping reason: {stop_reason}")
    print("Naive interpretation: The treatment appears harmful.")
    print(
        "Correct interpretation: Sequential monitoring can protect users by stopping harmful "
        "experiments early with a pre-planned rule."
    )


def run_section_4(n_experiments_for_main=300):
    """Compare naive and sequential false positive rates."""
    print(
        f"NOTE: Estimating false positive rates with {n_experiments_for_main} experiments "
        "in __main__ for runtime; function default remains 500."
    )
    rates = compare_false_positive_rates(n_experiments=n_experiments_for_main)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [
        "Naive daily peeking",
        "Sequential alpha-spending",
        "Intended alpha = 5%",
    ]
    values = [
        rates["naive_false_positive_rate"] * 100,
        rates["sequential_false_positive_rate"] * 100,
        rates["alpha"] * 100,
    ]
    ax.bar(labels, values)
    ax.set_title("False Positive Rate: Naive Peeking vs. Sequential Testing")
    ax.set_ylabel("False positive rate (%)")
    ax.set_ylim(0, max(values) * 1.3)
    fig.tight_layout()
    save_figure(fig, "m4_false_positive_comparison.png")

    print("\nSECTION 4 — FALSE POSITIVE RATE COMPARISON")
    print(f"Naive peeking false positive rate: {rates['naive_false_positive_rate'] * 100:.1f}%")
    print(
        "Sequential testing false positive rate: "
        f"{rates['sequential_false_positive_rate'] * 100:.1f}%"
    )
    print("Target alpha: 5.0%")
    print(
        "Correct interpretation: Sequential testing reduces false positives by using stricter "
        "per-look thresholds instead of using p < 0.05 at every look."
    )


def main():
    run_section_1()
    run_section_2()
    run_section_3()
    run_section_4()


if __name__ == "__main__":
    main()
