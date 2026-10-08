"""Shared datasets for the scikit-learn 1.8 and 1.9 notebooks."""

import numpy as np
import pandas as pd


def load_bank_marketing_data():
    """Load the UCI Bank Marketing features and target into one dataframe."""
    from ucimlrepo import fetch_ucirepo

    bank_marketing = fetch_ucirepo(id=222)
    return pd.concat(
        [bank_marketing.data.features, bank_marketing.data.targets], axis=1
    )


def load_penguins_data():
    """Load the curated Palmer Penguins dataset."""
    url = (
        "https://raw.githubusercontent.com/allisonhorst/"
        "palmerpenguins/main/inst/extdata/penguins.csv"
    )
    return pd.read_csv(url)


def make_customer_spending_data(
    n_customers=300, missing_rate=0.15, random_state=42
):
    """Create a regression dataset with missing feature values."""
    rng = np.random.default_rng(random_state)

    customer_data = pd.DataFrame(
        {
            "annual_income": rng.normal(60_000, 15_000, n_customers),
            "debt_ratio": rng.uniform(0, 1, n_customers),
            "years_with_bank": rng.integers(0, 25, n_customers).astype(float),
        }
    )

    annual_spending = (
        0.0003 * customer_data["annual_income"]
        - 8 * customer_data["debt_ratio"]
        + 0.4 * customer_data["years_with_bank"]
        + rng.normal(0, 2, n_customers)
    )

    missing_mask = rng.random(customer_data.shape) < missing_rate
    customer_data = customer_data.mask(missing_mask)

    return customer_data, annual_spending
