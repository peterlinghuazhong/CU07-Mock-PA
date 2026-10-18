"""Checker"""
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


def personal_values(L4, cfg):
    d = [int(c) for c in L4]
    pv = {"SEED": int(L4),
          "TEST_SIZE": 0.20 if d[3] % 2 == 0 else 0.25,
          "DEPTH": 3 + d[1] % 4,
          "PORT": 5100 + int(L4[2:]),
          "N_EST": cfg.get("n_estimators", 100)}
    if cfg["kind"] == "reg":
        pv["P1"] = d[2] + 1                      # Ridge alpha
        pv["P2"] = 10 * pv["P1"]                 # exam only
    else:
        pv["P1"] = round(0.04 * (d[2] + 1), 2)   # LogisticRegression C
        pv["P2"] = round(pv["P1"] * 10, 2)       # exam only
    return pv


def candidates(pv, cfg):
    from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    S, D, N = pv["SEED"], pv["DEPTH"], pv["N_EST"]
    rf_params = {"n_estimators": N, "max_depth": D}
    if cfg["kind"] == "reg":
        out = [("linear-base", LinearRegression(), {}, "linear"),
               (f"ridge-a{pv['P1']:g}", Ridge(alpha=pv["P1"]), {"alpha": pv["P1"]}, "linear")]
        if cfg["mode"] == "exam":
            out.append((f"ridge-a{pv['P2']:g}", Ridge(alpha=pv["P2"]), {"alpha": pv["P2"]}, "linear"))
        out.append((f"rf-d{D}", RandomForestRegressor(n_estimators=N, max_depth=D, random_state=S), rf_params, "tree_ensemble"))
    else:
        out = [("logreg-base", LogisticRegression(max_iter=1000), {"max_iter": 1000}, "linear"),
               (f"logreg-c{pv['P1']:g}", LogisticRegression(C=pv["P1"], max_iter=1000), {"C": pv["P1"], "max_iter": 1000}, "linear")]
        if cfg["mode"] == "exam":
            out.append((f"logreg-c{pv['P2']:g}", LogisticRegression(C=pv["P2"], max_iter=1000), {"C": pv["P2"], "max_iter": 1000}, "linear"))
        out.append((f"rf-d{D}", RandomForestClassifier(n_estimators=N, max_depth=D, random_state=S), rf_params, "tree_ensemble"))
    return out


def make_pipeline(model, cat_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
    return Pipeline([("prep", pre), ("model", model)])


def metric_dict(kind, y_true, y_pred):
    from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                                 accuracy_score, precision_score, recall_score, f1_score)
    if kind == "reg":
        return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred)), "f1": float(f1_score(y_true, y_pred))}


def rank_rule(L4, cfg):
    d4 = int(L4[3])
    exam = cfg["mode"] == "exam"
    if cfg["kind"] == "reg":
        return ("test_rmse" if (not exam or d4 % 2 == 0) else "test_mae"), True       # key, ascending
    return ("test_f1" if (not exam or d4 % 2 == 0) else "test_recall"), False


def gap_key(cfg):
    return "rmse" if cfg["kind"] == "reg" else "f1"


def pick_champion(scores, gaps, asc, exam, rel=0.01):
    """scores: {run: value of the ranking metric}; exam => tie rule (within rel of the best -> smaller overfit_gap)."""
    order = sorted(scores, key=lambda r: (scores[r] if asc else -scores[r], r))
    if not exam:
        return order[0]
    best = scores[order[0]]
    band = [r for r in order if abs(scores[r] - best) <= rel * abs(best)]
    return sorted(band, key=lambda r: (gaps[r], r))[0]


def oracle(L4, cfg):
    from sklearn.model_selection import train_test_split
    mine = build_my_data(L4, cfg)
    csv_text = mine.to_csv(index=False)
    df = pd.read_csv(io.StringIO(csv_text))
    pv = personal_values(L4, cfg)
    X = df.drop(columns=[cfg["id"], cfg["target"]]); y = df[cfg["target"]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=pv["TEST_SIZE"], random_state=pv["SEED"])
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    row = X.iloc[[int(L4) % len(X)]]
    runs = {}
    for name, model, params, mtype in candidates(pv, cfg):
        pipe = make_pipeline(model, cat_cols).fit(Xtr, ytr)
        tr = metric_dict(cfg["kind"], ytr, pipe.predict(Xtr)); te = metric_dict(cfg["kind"], yte, pipe.predict(Xte))
        g = gap_key(cfg)
        r = {"params": {"test_size": pv["TEST_SIZE"], "split_seed": pv["SEED"], **params}, "model_type": mtype,
             "train": tr, "test": te, "gap": abs(tr[g] - te[g])}
        r["row"] = float(pipe.predict(row)[0]) if cfg["kind"] == "reg" else float(pipe.predict_proba(row)[0, 1])
        runs[f"{name}-{L4}"] = r
    key, asc = rank_rule(L4, cfg)
    exam = cfg["mode"] == "exam"
    scores = {n: r["test"][key[5:]] for n, r in runs.items()}
    gaps = {n: r["gap"] for n, r in runs.items()}
    order = sorted(scores, key=lambda r: (scores[r] if asc else -scores[r], r))
    champion = pick_champion(scores, gaps, asc, exam)
    accepted = {pick_champion(scores, gaps, asc, exam, rel) for rel in (0.0095, 0.01, 0.0105)}
    if exam:   # numerical noise between machines: runs in the 1 % band whose gap is practically the winner's are also accepted
        best = scores[order[0]]
        band = [r for r in scores if abs(scores[r] - best) <= 0.01 * abs(best)]
        gmin = min(gaps[r] for r in band)
        accepted |= {r for r in band if gaps[r] <= gmin + max(5e-4, 0.02 * gmin)}
    return {"L4": L4, "pv": pv, "runs": runs, "key": key, "asc": asc, "scores": scores, "gaps": gaps, "order": order,
            "champion": champion, "accepted": accepted, "baseline": f"{candidates(pv, cfg)[0][0]}-{L4}",
            "row": row, "n_rows": len(df), "data_md5": hashlib.md5(csv_text.encode()).hexdigest()}


def challenger_for(O, champ):
    return [r for r in O["order"] if r != champ][0]


# --------------------------------------------------------------------------------------- checker
def _norm(name):
    return re.sub(r"\d+\.\d+", lambda m: f"{float(m.group()):g}", name or "")   # 'logreg-c4.0-0731' == 'logreg-c4-0731'


def _close(a, b, rel=0.02, abs_=1e-3):
    return abs(a - b) <= max(abs_, rel * abs(b))


def check(L4, cfg, reveal=False, workdir=".", serve=False):
    import mlflow
    from mlflow import MlflowClient
    O = oracle(L4, cfg)
    res = []
    def add(ok, text, hint=""):
        res.append((ok, text, hint))
    exam = cfg["mode"] == "exam"
    kind = cfg["kind"]
    dbname = f"{cfg['db']}_{L4}.db"
    dbpath = os.path.join(workdir, dbname)
    csvpath = os.path.join(workdir, cfg["file"])
    # data file
    if os.path.exists(csvpath):
        same = hashlib.md5(open(csvpath, "rb").read()).hexdigest() == O["data_md5"]
        add(same, f"Data file {cfg['file']} was made for IC ending {L4}",
            f"Run: python make_my_data.py {L4}  (the data file in the folder belongs to another IC)" if not same else "")
    else:
        add(None, f"Data file {cfg['file']} not found in the folder (cannot confirm it matches IC {L4})")
    if not os.path.exists(dbpath):
        add(False, f"Tracking store {dbname} found", f"Expected sqlite:///{dbname}. Check the name and that the L4 is correct.")
        return res, O
    tmp = tempfile.mkdtemp(); shutil.copy(dbpath, os.path.join(tmp, "s.db"))
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(tmp, 's.db')}")
    cl = MlflowClient()
    add(True, f"Tracking store {dbname} found")
    exp = cl.get_experiment_by_name(f"{cfg['exp']}_{L4}")
    if exp is None:
        add(False, f"Experiment {cfg['exp']}_{L4} exists", "Experiment name must be exactly <prefix>_<L4>.")
        return res, O
    add(True, f"Experiment {cfg['exp']}_{L4} exists")
    runs = [r for r in cl.search_runs([exp.experiment_id], max_results=1000) if r.info.status == "FINISHED"]
    byname = {}
    for r in runs: byname.setdefault(_norm(r.data.tags.get("mlflow.runName", "")), []).append(r)
    exp_names = list(O["runs"])
    add(len(runs) == len(exp_names), f"Exactly {len(exp_names)} finished runs in the experiment (found {len(runs)})",
        "Extra or missing runs. Use a fresh store and run the notebook once from top to bottom." if len(runs) != len(exp_names) else "")
    for n in exp_names:
        rs = byname.get(n, [])
        if len(rs) != 1:
            add(False, f"Run '{n}' present once (found {len(rs)})",
                f"Run names must be <candidate>-{L4}, using YOUR personal values." + (f" Expected name: {n}" if reveal else ""))
            continue
        r = rs[0]; E = O["runs"][n]
        bad = []
        for k, v in E["params"].items():
            got = r.data.params.get(k)
            try: okp = got is not None and abs(float(got) - float(v)) < 1e-9
            except Exception: okp = False
            if not okp: bad.append(f"{k} (got {got}" + (f", expected {v:g}" if reveal else "") + ")")
        if "model" not in r.data.params: bad.append("model")
        add(not bad, f"{n}: params correct", ("Check params: " + "; ".join(bad)) if bad else "")
        add(r.data.tags.get("model_type") == E["model_type"], f"{n}: tag model_type = {E['model_type']}")
        need = [f"{s}_{m}" for s in ("train", "test") for m in E["train"]] + (["overfit_gap"] if exam else [])
        miss = [m for m in need if m not in r.data.metrics]
        add(not miss, f"{n}: all required metrics logged", ("Missing: " + ", ".join(miss)) if miss else "")
        bm = []
        for s in ("train", "test"):
            for m, v in E[s].items():
                got = r.data.metrics.get(f"{s}_{m}")
                if got is not None and not _close(got, v, 0.02): bm.append(f"{s}_{m}={got:.4g}" + (f" (expected {v:.4g})" if reveal else ""))
        add(not bm, f"{n}: metric values match a correct run on YOUR split and data",
            ("Different numbers: " + "; ".join(bm) + ". Check seed, test_size, the data file and the pipeline.") if bm else "")
        if exam and "overfit_gap" in r.data.metrics:
            add(_close(r.data.metrics["overfit_gap"], E["gap"], 0.05, 2e-3), f"{n}: overfit_gap correct",
                f"got {r.data.metrics['overfit_gap']:.4g}" + (f", expected {E['gap']:.4g}" if reveal else ""))
    try:
        lms = cl.search_logged_models(experiment_ids=[exp.experiment_id])
        have = {m.source_run_id for m in lms}
        add(all(rs[0].info.run_id in have for rs in byname.values() if len(rs) == 1), "Every run has its model logged (log_model)")
    except Exception as e:
        add(None, f"Could not list logged models ({type(e).__name__})")

    # ---------------- registry
    reg = f"{cfg['reg']}_{L4}"
    def run_of(mv):
        if mv.run_id: return mv.run_id
        try:
            src = mv.source or ""
            if "m-" in src: return cl.get_logged_model(src.rstrip("/").split("/")[-1]).source_run_id
        except Exception: pass
        return None
    id2name = {r.info.run_id: _norm(r.data.tags.get("mlflow.runName")) for r in runs}
    try:
        mvs = cl.search_model_versions(f"name='{reg}'")
    except Exception:
        mvs = []
    if not mvs:
        add(False, f"Registered model {reg} with versions", "Registered-model name must be exactly <prefix>_<L4>.")
        return res, O
    mvs = sorted(mvs, key=lambda m: int(m.version))
    add(len(mvs) >= 2, f"Registered model {reg} has at least 2 versions (found {len(mvs)})")
    v1 = mvs[0]
    add(id2name.get(run_of(v1)) == O["baseline"], f"Version 1 is the baseline run ({O['baseline']})")
    try:
        ch = cl.get_model_version_by_alias(reg, "champion")
    except Exception:
        ch = None
    cn = None
    if ch is None:
        add(False, "Alias 'champion' exists")
    else:
        cn = id2name.get(run_of(ch))
        okc = cn in O["accepted"]
        add(okc, "Alias 'champion' points to the correct best run" + (f" (champion = {cn})" if cn else ""),
            ("Apply the ranking rule of YOUR personal values" + (f"; expected {sorted(O['accepted'])}" if reveal else "")) if not okc else "")
        add(int(ch.version) >= 2, "Champion is a version registered after the baseline (version 2 or later)")
        add(bool(ch.tags.get("approved_by")), "Champion version has tag approved_by")
        add(bool(ch.description), "Champion version has a description")
        if exam:
            chal_exp = challenger_for(O, cn) if cn in O["runs"] else None
            try:
                cb = cl.get_model_version_by_alias(reg, "challenger"); cbn = id2name.get(run_of(cb))
            except Exception:
                cb, cbn = None, None
            add(cb is not None and cbn == chal_exp, "Alias 'challenger' points to the best run other than the champion" + (f" (found {cbn})" if cbn else ""),
                ("expected " + str(chal_exp)) if reveal and cbn != chal_exp else "")
            tagname = "my_row_prediction" if kind == "reg" else "my_row_proba"
            tv = ch.tags.get(tagname)
            try:
                tvf = float(tv); ev = O["runs"][cn]["row"]
                okt = _close(tvf, ev, 0.01, 5.0) if kind == "reg" else abs(tvf - ev) <= 0.015
            except Exception:
                okt, ev = False, None
            add(okt, f"Champion version has tag {tagname} matching YOUR row",
                (f"got {tv}" + (f", expected about {ev:.4g}" if reveal and ev is not None else "")) if not okt else "")
    # ---------------- optional live serving check
    if serve:
        import requests
        try:
            payload = {"dataframe_split": json.loads(O["row"].to_json(orient="split", index=False))}
            r = requests.post(f"http://127.0.0.1:{O['pv']['PORT']}/invocations", json=payload, timeout=10)
            if r.status_code == 200 and cn in O["runs"]:
                got = float(np.array(r.json()["predictions"]).ravel()[0]); ev = O["runs"][cn]["row"]
                if kind == "cls": add(True, f"Server on port {O['pv']['PORT']} answered 200 (predicted class {got:g})")
                else: add(_close(got, ev, 0.01, 5.0), f"Server on port {O['pv']['PORT']} answered 200 with the expected prediction for YOUR row", f"got {got:.1f}")
            else:
                add(r.status_code == 200, f"Server on port {O['pv']['PORT']} answered {r.status_code}")
        except Exception:
            add(None, f"No server found on port {O['pv']['PORT']} (start it first to run this check)")
    shutil.rmtree(tmp, ignore_errors=True)
    return res, O


def report(res, O, title):
    print("=" * 70); print(title); print("=" * 70)
    ok = sum(1 for r in res if r[0] is True); bad = sum(1 for r in res if r[0] is False); warn = sum(1 for r in res if r[0] is None)
    for good, text, hint in res:
        print(("  [ OK ] " if good is True else "  [FAIL] " if good is False else "  [ ?? ] ") + text)
        if hint and good is not True: print("         -> " + hint)
    print("-" * 70); print(f"  {ok} passed, {bad} failed, {warn} could not be checked"); return ok, bad, warn


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Self-check for the mock PA: compares your MLflow store with the expected results for YOUR IC.")
    ap.add_argument("L4", help="last four digits of the IC")
    ap.add_argument("--folder", default=".", help="folder that holds the data file and the .db store (default: here)")
    ap.add_argument("--serve", action="store_true", help="also call the model server on your port (start it first)")
    pass
    a = ap.parse_args()
    pass
    res, O = check(a.L4, CFG, reveal=True, workdir=a.folder, serve=a.serve)
    report(res, O, 'MOCK self-check – PinjamPantas' + "  (IC ending " + a.L4 + ")")
    if O:
        print("  Personal values:", {k: v for k, v in O["pv"].items() if k != "N_EST"})
        print("  Ranking metric:", O["key"], "| expected champion:", sorted(O["accepted"]))
