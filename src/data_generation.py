"""Synthetic subscription-customer churn dataset generator.

One row per customer (a monthly snapshot), with churn probability driven by a
mix of tenure, contract type, support-ticket volume, usage, and payment method,
so the notebook has real (if synthetic) structure to model and explain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
CONTRACT_WEIGHTS = [0.55, 0.28, 0.17]

PAYMENT_METHODS = ["credit-card", "bank-transfer", "e-check", "mailed-check"]
PAYMENT_WEIGHTS = [0.35, 0.25, 0.28, 0.12]


def generate_customers(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST_{i:06d}" for i in range(1, n + 1)]
    tenure_months = rng.integers(1, 73, size=n)
    contract_type = rng.choice(CONTRACT_TYPES, size=n, p=CONTRACT_WEIGHTS)
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=PAYMENT_WEIGHTS)

    monthly_charges = np.round(rng.normal(65, 20, size=n).clip(15, 150), 2)
    # total_charges derived from tenure * monthly_charges with noise, mimicking a
    # real billing-history rollup rather than an independently invented column
    total_charges = np.round(
        tenure_months * monthly_charges * rng.normal(1.0, 0.05, size=n), 2
    ).clip(0, None)

    num_support_tickets = rng.poisson(1.2, size=n)
    avg_monthly_usage = rng.normal(55, 20, size=n).clip(0, 100)
    has_addon_service = rng.random(n) < 0.35
    auto_pay_enabled = rng.random(n) < 0.55

    # --- churn probability model (ground truth we then sample a label from) ---
    logit = -1.2
    logit += np.where(contract_type == "month-to-month", 1.4, 0.0)
    logit += np.where(contract_type == "one-year", 0.2, 0.0)
    logit += np.where(payment_method == "e-check", 0.5, 0.0)
    logit += -0.03 * tenure_months
    logit += 0.35 * num_support_tickets
    logit += -0.02 * avg_monthly_usage
    logit += np.where(has_addon_service, -0.3, 0.0)
    logit += np.where(auto_pay_enabled, -0.25, 0.0)
    logit += 0.01 * (monthly_charges - 65)
    logit += rng.normal(0, 0.6, size=n)  # unexplained noise

    churn_prob = 1 / (1 + np.exp(-logit))
    churned = rng.random(n) < churn_prob

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "avg_monthly_usage": np.round(avg_monthly_usage, 1),
            "has_addon_service": has_addon_service,
            "auto_pay_enabled": auto_pay_enabled,
            "churned": churned,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_customers()
    print(df["churned"].mean())
    df.to_csv("data/customers.csv", index=False)
