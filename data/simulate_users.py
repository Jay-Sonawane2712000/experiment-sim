"""Synthetic user-level data generation for experimentation examples."""

import numpy as np
import pandas as pd


def simulate_users(n_users, seed, effect_size):
    """Simulate e-commerce users for A/B testing failure demonstrations."""
    rng = np.random.default_rng(seed)

    # Sequential integer IDs keep user-level joins and debugging simple in later modules.
    user_id = np.arange(1, n_users + 1)

    # A 50/50 split maximizes statistical power for a fixed experiment population.
    group = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])

    # Log-normal location reflects typical e-commerce revenue centered around tens of dollars.
    revenue_mu = 3.5

    # High log-normal spread captures the right-skew caused by a small number of high-value buyers.
    revenue_sigma = 1.2

    # Pre-period revenue is used later for variance reduction and imbalance diagnostics.
    pre_experiment_revenue = rng.lognormal(
        mean=revenue_mu,
        sigma=revenue_sigma,
        size=n_users,
    )

    # The control post-period represents the counterfactual baseline purchase behavior.
    baseline_post_revenue = rng.lognormal(
        mean=revenue_mu,
        sigma=revenue_sigma,
        size=n_users,
    )

    # Treatment lift is modeled multiplicatively because product changes often affect spend proportionally.
    treatment_multiplier = 1 + effect_size

    # Additional post-period noise reflects week-to-week purchase volatility unrelated to treatment.
    treatment_noise = rng.lognormal(
        mean=0.0,
        sigma=0.2,
        size=n_users,
    )

    post_experiment_revenue = np.where(
        group == "treatment",
        baseline_post_revenue * treatment_multiplier * treatment_noise,
        baseline_post_revenue,
    )

    # Most users use one device, while a meaningful minority create cross-device contamination risk.
    single_device_probability = 0.9

    # Multi-device users are capped at a small practical range for clear contamination examples.
    max_extra_devices = 3

    device_count = np.where(
        rng.random(n_users) < single_device_probability,
        1,
        rng.integers(2, max_extra_devices + 1, size=n_users),
    )

    return pd.DataFrame(
        {
            "user_id": user_id,
            "group": group,
            "pre_experiment_revenue": pre_experiment_revenue,
            "post_experiment_revenue": post_experiment_revenue,
            "device_count": device_count,
        }
    )


if __name__ == "__main__":
    users = simulate_users(n_users=10000, seed=42, effect_size=0.05)

    print("First 5 rows:")
    print(users.head())

    print("\nGroup sizes:")
    print(users["group"].value_counts())

    print("\nMean post_experiment_revenue per group:")
    print(users.groupby("group")["post_experiment_revenue"].mean())

    print("\nDistributional summary for post_experiment_revenue:")
    print(users["post_experiment_revenue"].describe())

    group_sizes = users["group"].value_counts()
    size_difference = abs(group_sizes["control"] - group_sizes["treatment"])
    allowed_difference = 0.05 * len(users)

    if size_difference > allowed_difference:
        print("\nWARNING: Control and treatment groups differ by more than 5%.")
    else:
        print("\nGroup sizes are within 5% of each other.")
