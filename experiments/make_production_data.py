"""Create a labelled 'production' file for the churn sample with KNOWN, documented changes.

Simulated shift (clearly artificial - report it as such):
  data drift     more month-to-month contracts (55% -> 75%), monthly charges +15,
                 more fibre internet (45% -> 65%)
  concept drift  from row 400 on, fibre customers churn much more for the same inputs
Rows are in time order, so a concept-drift detector can find the change point.
"""
import numpy as np
import pandas as pd

SEED, N, CHANGE_AT = 7, 1000, 400
rng = np.random.default_rng(SEED)

tenure = rng.integers(0, 72, N)
charges = (rng.normal(65, 30, N) + 15).clip(18, 140).round(2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[.75, .15, .10])
internet = rng.choice(["DSL", "Fiber optic", "No"], N, p=[.25, .65, .10])
logit = -1 + 1.5 * (contract == "Month-to-month") - 0.04 * tenure + 0.02 * (charges - 65) \
        + 0.5 * (internet == "Fiber optic")
after = np.arange(N) >= CHANGE_AT
logit = logit + np.where(after & (internet == "Fiber optic"), 2.5, 0.0)   # concept change
churn = np.where(rng.random(N) < 1 / (1 + np.exp(-logit)), "Yes", "No")

df = pd.DataFrame({
    "customer_id": np.arange(20001, 20001 + N),
    "gender": rng.choice(["Male", "Female"], N),
    "senior_citizen": rng.choice([0, 1], N, p=[.84, .16]),
    "tenure_months": tenure, "contract": contract, "internet_service": internet,
    "payment_method": rng.choice(["UPI", "Credit card", "Bank transfer", "Cash"], N),
    "monthly_charges": charges, "total_charges": (charges * tenure).round(2),
    "signup_date": (pd.Timestamp("2021-06-01") + pd.to_timedelta(rng.integers(0, 1800, N), unit="D"))
    .strftime("%Y-%m-%d"),
    "churn": churn,
})
df.to_csv("datasets/sample/churn_production.csv", index=False)
print(f"Saved datasets/sample/churn_production.csv ({N} rows, concept change at row {CHANGE_AT})")
