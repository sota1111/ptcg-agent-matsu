"""Promoted Matsu stability profile bundled with the live submission."""

PROFILE = {
    "schema_version": "ptcg-matsu-runtime-profile/v1",
    "profile_id": "matsu-stability-control-v1",
    "risk_profile": "conservative",
    "strategy": "resource-preserving-control",
    "source_issue": "SOT-1848",
    "promotion_issue": "SOT-1868",
    "search": {
        "time_budget_s": 0.25,
        "max_tree_depth": 5,
        "uct_c": 0.72,
        "deviate_margin": 0.28,
    },
    "evaluation": {
        "eval_weights": {
            "deck_low": -0.22,
            "deck_low_at": 12,
            "deck_low_prize_gate": 3,
        },
    },
    "fallback": "highest-value-legal",
}


def runtime_overrides():
    """Return a fresh PlannerConfig-compatible mapping for each agent."""
    values = dict(PROFILE["search"])
    values["eval_weights"] = dict(PROFILE["evaluation"]["eval_weights"])
    return values
