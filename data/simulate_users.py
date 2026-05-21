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
    propensity_mu = 3.5

    # User-level propensity captures persistent customer value: casual shoppers and loyal buyers.
    propensity_sigma = 1.0

    # CUPED requires a pre-experiment covariate that predicts experiment-period revenue. In
    # e-commerce, past spending is usually predictive of future spending, so we intentionally
    # simulate positive correlation between pre- and post-experiment revenue.
    user_spending_propensity = rng.lognormal(
        mean=propensity_mu,
        sigma=propensity_sigma,
        size=n_users,
    )

    # Moderate pre-period noise reflects normal purchase timing variation before the experiment.
    pre_noise_sigma = 0.55

    # Setting log-noise mean to -sigma^2 / 2 keeps the multiplier centered around 1 in dollars.
    pre_noise_mu = -(pre_noise_sigma**2) / 2

    # Pre-period revenue is used later for variance reduction and imbalance diagnostics.
    pre_experiment_revenue = user_spending_propensity * rng.lognormal(
        mean=pre_noise_mu,
        sigma=pre_noise_sigma,
        size=n_users,
    )

    # Moderate post-period noise keeps revenue realistic while preserving user-level predictability.
    post_noise_sigma = 0.55

    # Centered post noise keeps the baseline mean tied to long-run customer spending propensity.
    post_noise_mu = -(post_noise_sigma**2) / 2

    # The control post-period represents the counterfactual baseline purchase behavior.
    baseline_post_revenue = user_spending_propensity * rng.lognormal(
        mean=post_noise_mu,
        sigma=post_noise_sigma,
        size=n_users,
    )

    # Treatment lift is modeled multiplicatively because product changes often affect spend proportionally.
    treatment_multiplier = 1 + effect_size

    post_experiment_revenue = np.where(
        group == "treatment",
        baseline_post_revenue * treatment_multiplier,
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

    print("\nCorrelation between pre_experiment_revenue and post_experiment_revenue:")
    print(users["pre_experiment_revenue"].corr(users["post_experiment_revenue"]))

    group_sizes = users["group"].value_counts()
    size_difference = abs(group_sizes["control"] - group_sizes["treatment"])
    allowed_difference = 0.05 * len(users)

    if size_difference > allowed_difference:
        print("\nWARNING: Control and treatment groups differ by more than 5%.")
    else:
        print("\nGroup sizes are within 5% of each other.")
