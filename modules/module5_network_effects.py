"""Phase 5: network effects and interference in revenue A/B tests."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.simulate_users import simulate_users


FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def save_figure(fig, filename):
    """Save figures to the project output folder used for review and dashboarding."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_p_value(p_value):
    return f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"


def create_user_network(n_users=10000, seed=42):
    """Create a clustered user network with one node per user."""
    # Standard A/B tests assume one user's treatment does not affect another user's outcome. In real
    # products, users can influence each other through recommendations, shared purchases, family
    # accounts, or social behavior. This creates spillover effects and violates independence.
    # A connected Watts-Strogatz graph gives local neighborhoods and short paths, which is a simple
    # stand-in for household/community influence without adding new dependencies.
    return nx.connected_watts_strogatz_graph(
        n=n_users,
        k=8,
        p=0.08,
        tries=100,
        seed=seed,
    )


def _assign_individual_groups(n_users, seed):
    rng = np.random.default_rng(seed + 50_000)
    return rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])


def _assign_cluster_groups(graph, n_users, seed):
    rng = np.random.default_rng(seed + 60_000)
    assigned_group = np.empty(n_users, dtype=object)

    # Greedy modularity is a simple built-in community detector. It may return communities of
    # different sizes, but assigning each detected group together keeps the implementation readable
    # while demonstrating how cluster randomization reduces cross-group spillovers.
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    for community in communities:
        group = rng.choice(["control", "treatment"])
        assigned_group[list(community)] = group

    return assigned_group, communities


def _treated_neighbor_share(graph, is_treated):
    shares = np.zeros(len(is_treated))
    neighbor_counts = np.zeros(len(is_treated), dtype=int)
    treated_neighbor_counts = np.zeros(len(is_treated), dtype=int)

    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        neighbor_counts[node] = len(neighbors)
        if neighbors:
            treated_count = int(is_treated[neighbors].sum())
            treated_neighbor_counts[node] = treated_count
            shares[node] = treated_count / len(neighbors)

    return neighbor_counts, treated_neighbor_counts, shares


def simulate_network_experiment(
    n_users=10000,
    seed=42,
    treatment_effect=0.05,
    spillover_effect=0.02,
    assignment_type="individual",
):
    """Simulate direct treatment and neighbor spillover effects on revenue."""
    baseline_df = simulate_users(n_users=n_users, seed=seed, effect_size=0.0).copy()
    graph = create_user_network(n_users=n_users, seed=seed)

    if assignment_type == "individual":
        assigned_group = _assign_individual_groups(n_users, seed)
    elif assignment_type == "cluster":
        assigned_group, _ = _assign_cluster_groups(graph, n_users, seed)
    else:
        raise ValueError("assignment_type must be 'individual' or 'cluster'.")

    baseline_df["assigned_group"] = assigned_group
    is_treated = assigned_group == "treatment"
    neighbor_counts, treated_neighbor_counts, treated_neighbor_share = _treated_neighbor_share(
        graph,
        is_treated,
    )

    baseline_revenue = baseline_df["post_experiment_revenue"].to_numpy()
    observed_revenue = baseline_revenue.copy()

    # Direct lift models users who personally see the new recommendation algorithm.
    observed_revenue[is_treated] *= 1 + treatment_effect

    # Spillover models untreated users being indirectly affected by nearby treated users through
    # shared purchase behavior, recommendations, or household/community influence.
    control_mask = ~is_treated
    observed_revenue[control_mask] *= (
        1 + treated_neighbor_share[control_mask] * spillover_effect
    )

    baseline_df["number_of_neighbors"] = neighbor_counts
    baseline_df["number_of_treated_neighbors"] = treated_neighbor_counts
    baseline_df["treated_neighbor_share"] = treated_neighbor_share
    baseline_df["observed_revenue"] = observed_revenue
    baseline_df["assignment_type"] = assignment_type

    return baseline_df, graph


def estimate_treatment_effect(df):
    """Estimate treatment lift using the standard difference in observed means."""
    control = df.loc[df["assigned_group"] == "control", "observed_revenue"]
    treatment = df.loc[df["assigned_group"] == "treatment", "observed_revenue"]

    control_mean = control.mean()
    treatment_mean = treatment.mean()
    estimated_lift = (treatment_mean - control_mean) / control_mean
    p_value = stats.ttest_ind(control, treatment, equal_var=False).pvalue

    return {
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "estimated_lift": estimated_lift,
        "p_value": p_value,
    }


def _print_network_summary(df, graph):
    control_df = df[df["assigned_group"] == "control"]
    print(f"Number of users: {len(df)}")
    print(f"Number of graph edges: {graph.number_of_edges()}")
    print(f"Average degree: {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")
    print(f"Number of treated users: {(df['assigned_group'] == 'treatment').sum()}")
    print(f"Number of control users: {(df['assigned_group'] == 'control').sum()}")
    print(
        "Average treated-neighbor share among control users: "
        f"{control_df['treated_neighbor_share'].mean():.3f}"
    )


def run_section_1():
    """Simulate the graph and print network spillover summary information."""
    experiment_df, graph = simulate_network_experiment()

    print("\nSECTION 1 — SIMULATE SOCIAL GRAPH AND NETWORK SPILLOVERS")
    _print_network_summary(experiment_df, graph)
    print(
        "Correct interpretation: The graph creates connected user neighborhoods where treatment "
        "can spill over from treated users to nearby controls."
    )


def run_section_2():
    """Show bias from individual randomization under network interference."""
    # When control users are exposed to treated neighbors, the control group is no longer a clean
    # counterfactual. This can cause standard A/B testing to underestimate or overestimate the true
    # direct treatment effect.
    df, _ = simulate_network_experiment(
        n_users=10000,
        seed=42,
        treatment_effect=0.05,
        spillover_effect=0.03,
        assignment_type="individual",
    )
    estimates = estimate_treatment_effect(df)
    average_control_exposure = df.loc[
        df["assigned_group"] == "control", "treated_neighbor_share"
    ].mean()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = ["True direct\ntreatment effect", "Estimated lift under\nindividual randomization"]
    values = [5.0, estimates["estimated_lift"] * 100]
    ax.bar(labels, values)
    ax.set_title("Individual Randomization Is Biased Under Network Interference")
    ax.set_ylabel("Revenue lift (%)")
    ax.text(
        0.5,
        0.94,
        f"Avg treated-neighbor share among controls: {average_control_exposure:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    fig.tight_layout()
    save_figure(fig, "m5_individual_interference.png")

    naive_conclusion = (
        "The standard A/B test would conclude the treatment increases revenue."
        if estimates["p_value"] < 0.05
        else "The standard A/B test would not find a statistically significant lift."
    )

    print("\nSECTION 2 — INDIVIDUAL RANDOMIZATION UNDER INTERFERENCE")
    print("True direct treatment effect: 5.0%")
    print(
        "Estimated lift under individual randomization: "
        f"{estimates['estimated_lift'] * 100:.2f}%"
    )
    print(f"P-value: {format_p_value(estimates['p_value'])}")
    print(
        "Average treated-neighbor share among control users: "
        f"{average_control_exposure:.3f}"
    )
    print(f"Naive conclusion: {naive_conclusion}")
    print(
        "Correct interpretation: Control users are partially exposed through treated neighbors, "
        "so the control group is contaminated and the estimated treatment effect is biased."
    )

    return df, estimates


def run_section_3(individual_estimates):
    """Compare individual and cluster randomization under interference."""
    # Cluster randomization assigns connected users together, reducing spillover between treatment
    # and control. This creates a cleaner comparison when users influence each other.
    cluster_df, _ = simulate_network_experiment(
        n_users=10000,
        seed=42,
        treatment_effect=0.05,
        spillover_effect=0.03,
        assignment_type="cluster",
    )
    cluster_estimates = estimate_treatment_effect(cluster_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [
        "True direct\neffect",
        "Individual\nrandomization",
        "Cluster\nrandomization",
    ]
    values = [
        5.0,
        individual_estimates["estimated_lift"] * 100,
        cluster_estimates["estimated_lift"] * 100,
    ]
    ax.bar(labels, values)
    ax.set_title("Cluster Randomization Reduces Network Interference Bias")
    ax.set_ylabel("Revenue lift (%)")
    fig.tight_layout()
    save_figure(fig, "m5_cluster_correction.png")

    print("\nSECTION 3 — CLUSTER RANDOMIZATION CORRECTION")
    print("True direct treatment effect: 5.0%")
    print(
        "Estimated lift under individual randomization: "
        f"{individual_estimates['estimated_lift'] * 100:.2f}%"
    )
    print(
        "Estimated lift under cluster randomization: "
        f"{cluster_estimates['estimated_lift'] * 100:.2f}%"
    )
    print(f"Individual randomization p-value: {format_p_value(individual_estimates['p_value'])}")
    print(f"Cluster randomization p-value: {format_p_value(cluster_estimates['p_value'])}")
    print(
        "Correct interpretation: Cluster randomization reduces cross-group contamination by "
        "assigning connected users together, making the treatment comparison more defensible under "
        "network effects."
    )

    return cluster_df, cluster_estimates


def run_section_4():
    """Create a readable network visualization for stakeholder explanation."""
    # A visual network makes interference easier to explain to non-technical stakeholders. It shows
    # why treating one user can indirectly affect nearby control users.
    df, graph = simulate_network_experiment(
        n_users=300,
        seed=7,
        treatment_effect=0.05,
        spillover_effect=0.03,
        assignment_type="individual",
    )

    treated = df["assigned_group"] == "treatment"
    exposed_control = (df["assigned_group"] == "control") & (
        df["number_of_treated_neighbors"] > 0
    )

    node_colors = np.where(
        treated,
        "#1f77b4",
        np.where(exposed_control, "#ff7f0e", "#bdbdbd"),
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    positions = nx.spring_layout(graph, seed=7, k=0.2)
    nx.draw_networkx_edges(graph, positions, ax=ax, alpha=0.12, width=0.6)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=np.where(exposed_control, 42, 28),
        linewidths=np.where(exposed_control, 0.7, 0.0),
        edgecolors=np.where(exposed_control, "black", "none"),
    )
    ax.set_title("Network Interference: Control Users Exposed to Treated Neighbors")
    ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Treatment", markerfacecolor="#1f77b4", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Exposed control", markerfacecolor="#ff7f0e", markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Unexposed control", markerfacecolor="#bdbdbd", markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "m5_network_visualization.png")

    print("\nSECTION 4 — NETWORK VISUALIZATION")
    print("Visualization saved to outputs/figures/m5_network_visualization.png")
    print(
        "Correct interpretation: Some control users are adjacent to treated users, so they may "
        "experience indirect treatment exposure through spillover."
    )


def main():
    run_section_1()
    _, individual_estimates = run_section_2()
    run_section_3(individual_estimates)
    run_section_4()


if __name__ == "__main__":
    main()
