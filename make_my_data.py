"""Creates YOUR OWN data file for the PinjamPantas mock PA.  Usage:  python make_my_data.py <last four digits of your IC>"""
import sys

def master_loans():
    import numpy as np, pandas as pd
    rng = np.random.default_rng(20261012)
    N = 1500
    employment = rng.choice(["Salaried", "Self-employed", "Contract", "Unemployed"], N, p=[.50, .22, .20, .08])
    loan_purpose = rng.choice(["Car", "Home", "Education", "Personal"], N, p=[.30, .22, .18, .30])
    housing = rng.choice(["Rent", "Own", "Family"], N, p=[.38, .34, .28])
    region = rng.choice(["Central", "North", "South", "East"], N, p=[.42, .20, .22, .16])
    age = np.clip(rng.normal(38, 10, N), 21, 65).astype(int)
    late_payments = np.clip(rng.poisson(0.9, N), 0, 8)
    monthly_income_rm = np.round(np.clip(rng.lognormal(8.3, 0.45, N), 1500, 25000) * np.where(employment == "Unemployed", 0.4, 1.0), 0)
    loan_amount_rm = np.round(np.clip(rng.lognormal(9.8, 0.7, N), 2000, 250000), 0)
    burden = loan_amount_rm / (monthly_income_rm * 12 + 1)
    logit = 1.7 * (-2.75 + 0.55 * late_payments + 1.6 * np.minimum(burden, 3.0) - 0.00012 * (monthly_income_rm - 4000)
                   + np.select([employment == "Unemployed", employment == "Contract", employment == "Self-employed"], [1.2, 0.45, 0.30], 0.0)
                   + np.select([loan_purpose == "Personal", loan_purpose == "Education"], [0.40, -0.20], 0.0)
                   + np.select([housing == "Rent", housing == "Own"], [0.30, -0.35], 0.0)
                   - 0.018 * (age - 38)
                   + 0.70 * ((late_payments >= 3) & (burden > 1.0))
                   + rng.normal(0, 0.20, N))
    defaulted = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"applicant_id": [f"A{i:04d}" for i in range(1, N + 1)],
                         "age": age, "late_payments": late_payments, "monthly_income_rm": monthly_income_rm, "loan_amount_rm": loan_amount_rm,
                         "employment": employment, "loan_purpose": loan_purpose, "housing": housing, "region": region,
                         "defaulted": defaulted})

MASTER = master_loans
CFG = {'kind': 'cls', 'mode': 'mock', 'id': 'applicant_id', 'target': 'defaulted', 'file': 'loan_applicants.csv', 'brand': 'PinjamPantas', 'exp': 'PinjamPantas_Default', 'reg': 'pinjampantas_default', 'db': 'pinjampantas', 'n_rows': 1200, 'n_estimators': 100}


# Needs a global MASTER (the scenario's master-data function) and a dict CFG.
import io, os, re, sys, json, shutil, tempfile, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def build_my_data(L4, cfg):
    """The candidate's own copy of the data: a seeded sample of rows and a shuffled column order."""
    df = MASTER()
    rng = np.random.default_rng(int(L4) + 20261007)
    idx = np.sort(rng.choice(len(df), cfg["n_rows"], replace=False))
    df = df.iloc[idx].reset_index(drop=True)
    feats = [c for c in df.columns if c not in (cfg["id"], cfg["target"])]
    order = [feats[i] for i in rng.permutation(len(feats))]
    return df[[cfg["id"]] + order + [cfg["target"]]]




if __name__ == "__main__":
    if len(sys.argv) != 2 or not (len(sys.argv[1]) == 4 and sys.argv[1].isdigit()):
        sys.exit("Usage: python make_my_data.py <last four digits of your IC>   e.g. python make_my_data.py 0731")
    L4 = sys.argv[1]
    mine = build_my_data(L4, CFG)
    mine.to_csv(CFG["file"], index=False)
    print(f"Created {CFG['file']} for IC ending {L4}: {mine.shape[0]} rows x {mine.shape[1]} columns.")
    print("Use THIS file in your notebook. Do not share it: another candidate's file is different.")
