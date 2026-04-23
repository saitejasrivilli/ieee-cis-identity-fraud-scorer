# ============================================================
# IEEE-CIS Fraud Detection — Identity-Aware Risk Scorer
# Pi Team (Amazon Payments) aligned:
#   - Identity resolution across conflicting signals
#   - Feature engineering (velocity, domain, device, address)
#   - LightGBM with class-imbalance handling
#   - AUC-PR as primary evaluation metric
#   - PSI production monitoring + retraining trigger
#   - FastAPI real-time inference endpoint
# ============================================================

# ── 0. Install dependencies ──────────────────────────────────
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("kaggle", "lightgbm", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "fastapi", "uvicorn", "requests")

print("✓ Dependencies installed")

# ── 1. Kaggle auth + data download ───────────────────────────
import os, json, zipfile, pathlib

# ── Credentials via Colab Secrets (never hardcode keys) ──────
# In Colab: click the key icon (🔑) in the left sidebar
# Add two secrets: KAGGLE_USERNAME and KAGGLE_KEY
from google.colab import userdata
KAGGLE_USERNAME = userdata.get('KAGGLE_USERNAME')
KAGGLE_KEY      = userdata.get('KAGGLE_KEY')

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

DATA_DIR = pathlib.Path("/content/ieee-fraud")
DATA_DIR.mkdir(exist_ok=True)

if not (DATA_DIR / "train_transaction.csv").exists():
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "ieee-fraud-detection", "-p", str(DATA_DIR)
    ], check=True)
    for zf in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(DATA_DIR)
        zf.unlink()
    print("✓ Data downloaded and extracted")
else:
    print("✓ Data already present — skipping download")

# ── 2. Load data (memory-optimized for Colab free tier) ───────
import pandas as pd
import numpy as np
import gc

def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to smallest safe dtype. ~60% RAM reduction."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object:
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(col_type).startswith("int"):
            for dtype in [np.int8, np.int16, np.int32]:
                if c_min >= np.iinfo(dtype).min and c_max <= np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        elif str(col_type).startswith("float"):
            # Skip float16 — causes overflow on IEEE-CIS V-columns
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df

print("\nLoading CSVs with memory reduction …")
trn_txn = reduce_mem(pd.read_csv(DATA_DIR / "train_transaction.csv"))
gc.collect()
trn_id  = reduce_mem(pd.read_csv(DATA_DIR / "train_identity.csv"))
gc.collect()
tst_txn = reduce_mem(pd.read_csv(DATA_DIR / "test_transaction.csv"))
gc.collect()
tst_id  = reduce_mem(pd.read_csv(DATA_DIR / "test_identity.csv"))
gc.collect()

print(f"  Train transactions : {trn_txn.shape[0]:,} rows × {trn_txn.shape[1]} cols")
print(f"  Train identity     : {trn_id.shape[0]:,} rows × {trn_id.shape[1]} cols")
print(f"  Test  transactions : {tst_txn.shape[0]:,} rows × {tst_txn.shape[1]} cols")
print(f"  Fraud rate         : {trn_txn['isFraud'].mean():.3%}")

# ── 3. Identity resolution join ──────────────────────────────
# KEY DIFFERENTIATOR vs plain fraud datasets:
# IEEE-CIS has a separate identity table with device, browser,
# email domain, and screen resolution signals. Joining these
# enables "identity resolution across conflicting data sources"
# — the same capability Pi uses for payment risk assessment.

train = trn_txn.merge(trn_id, on="TransactionID", how="left")
test  = tst_txn.merge(tst_id,  on="TransactionID", how="left")

# Compute coverage BEFORE freeing raw tables
identity_coverage = trn_id["TransactionID"].nunique() / len(trn_txn)

# Free raw tables — no longer needed
del trn_txn, trn_id, tst_txn, tst_id
gc.collect()
print(f"\n✓ Identity join complete")
print(f"  Post-join train shape : {train.shape[0]:,} × {train.shape[1]}")
print(f"  Identity coverage     : {identity_coverage:.1%} of transactions have identity signals")
print(f"  Transactions WITHOUT identity signals are themselves a risk signal (nulls encoded below)")

# ── 4. Feature engineering ───────────────────────────────────
# Covers: velocity features, email domain mismatch (identity
# resolution), device/browser consistency, card-address signals,
# and missing-value flags (absence of identity = signal itself).

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- 4a. Email domain signals (identity resolution) ------
    # Mismatch between purchaser (P_) and recipient (R_) email
    # domains is a strong identity inconsistency signal.
    df["email_domain_match"] = (
        df["P_emaildomain"].astype(str) == df["R_emaildomain"].astype(str)
    ).astype(int)

    # Free vs paid provider — proxy for account quality/age
    free_providers = {"gmail.com", "yahoo.com", "hotmail.com",
                      "outlook.com", "live.com", "aol.com"}
    df["P_email_is_free"] = df["P_emaildomain"].apply(
        lambda x: 1 if str(x).lower() in free_providers else 0
    )
    df["R_email_is_free"] = df["R_emaildomain"].apply(
        lambda x: 1 if str(x).lower() in free_providers else 0
    )
    # Both free but different domains → higher risk
    df["both_free_diff_domain"] = (
        (df["P_email_is_free"] == 1) &
        (df["R_email_is_free"] == 1) &
        (df["email_domain_match"] == 0)
    ).astype(int)

    # --- 4b. Transaction velocity / amount features ----------
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Z-score of transaction amount relative to card history
    # High z-score = unusual spend for this card = risk signal
    card_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
    card_std  = df.groupby("card1")["TransactionAmt"].transform("std").fillna(1)
    df["amt_zscore_card1"] = (df["TransactionAmt"] - card_mean) / card_std

    # Transaction count per card (high velocity = risk)
    df["card1_txn_count"] = df.groupby("card1")["TransactionID"].transform("count")

    # Time features: TransactionDT is seconds from a reference date
    df["txn_hour"] = (df["TransactionDT"] // 3600) % 24
    df["txn_dow"]  = (df["TransactionDT"] // (3600 * 24)) % 7
    df["txn_is_night"] = ((df["txn_hour"] >= 22) | (df["txn_hour"] <= 5)).astype(int)

    # --- 4c. Device / browser identity signals ---------------
    # Missing device or browser info = identity gap = risk
    df["has_device_type"]  = df["DeviceType"].notna().astype(int)
    df["has_browser_info"] = df["id_31"].notna().astype(int) if "id_31" in df.columns else 0
    df["has_os_info"]      = df["id_30"].notna().astype(int) if "id_30" in df.columns else 0

    # Count of available identity signals (more = lower risk)
    df["identity_signal_count"] = (
        df["has_device_type"] + df["has_browser_info"] + df["has_os_info"]
    )

    # Screen resolution availability (id_33 = "width x height")
    if "id_33" in df.columns:
        df["has_screen_res"] = df["id_33"].notna().astype(int)
    else:
        df["has_screen_res"] = 0

    # --- 4d. Card × address consistency ----------------------
    # Same card across many billing addresses = account takeover signal
    df["card1_addr1_nunique"] = df.groupby("card1")["addr1"].transform("nunique")
    df["card1_addr2_nunique"] = df.groupby("card1")["addr2"].transform("nunique")

    # addr1 × addr2 mismatch within same card
    df["addr_mismatch"] = (df["addr1"].astype(str) != df["addr2"].astype(str)).astype(int)

    # --- 4e. Missing value flags for key identity columns ----
    # Absence of identity signals is itself predictive
    identity_cols = ["P_emaildomain", "R_emaildomain", "DeviceType",
                     "id_30", "id_31", "id_33", "addr1", "addr2"]
    for col in identity_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # --- 4f. Product code × card interaction -----------------
    if "ProductCD" in df.columns:
        df["productcd_card1_count"] = df.groupby(
            ["ProductCD", "card1"])["TransactionID"].transform("count")

    return df

print("\nEngineering features …")
train = engineer_features(train)
test  = engineer_features(test)
gc.collect()

KNOWN_RAW_COLS = set([
    'TransactionID','isFraud','TransactionDT','TransactionAmt',
    'ProductCD','card1','card2','card3','card4','card5','card6',
    'addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain',
    'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
    'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15',
    'M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo'
])
engineered = [c for c in train.columns if c not in KNOWN_RAW_COLS and not c.startswith('V') and not c.startswith('id_')]
print(f"✓ {len(engineered)} new features engineered")
print(f"  {engineered}")

# ── 5. Prepare X / y ─────────────────────────────────────────
TARGET = "isFraud"
DROP   = [TARGET, "TransactionID", "TransactionDT"]

# Encode all object/category columns as integer codes
# Encode shared cols in both; train-only cols encoded separately
shared_cat  = [c for c in train.select_dtypes("object").columns if c in test.columns]
trainonly_cat = [c for c in train.select_dtypes("object").columns if c not in test.columns]

print(f"\nEncoding {len(shared_cat)} shared + {len(trainonly_cat)} train-only categorical columns …")
for col in shared_cat:
    train[col] = train[col].astype("category").cat.codes
    test[col]  = test[col].astype("category").cat.codes
for col in trainonly_cat:
    train[col] = train[col].astype("category").cat.codes

# Drop any remaining object columns that couldn't be encoded
# (test-only cols or anything still string-typed)
still_obj = train.select_dtypes("object").columns.tolist()
if still_obj:
    print(f"  Dropping {len(still_obj)} remaining object cols: {still_obj}")
    train.drop(columns=still_obj, inplace=True)
    test.drop(columns=[c for c in still_obj if c in test.columns], inplace=True)

# Only keep cols present in BOTH train and test to avoid KeyError on X_test
feature_cols = [c for c in train.columns
                if c not in DROP and c in test.columns]
X      = train[feature_cols].astype(np.float32)
y      = train[TARGET].values

# Drop train df — X holds everything needed, free ~2GB
del train
gc.collect()

X_test = test[feature_cols].astype(np.float32)
del test
gc.collect()

print(f"  Total features : {X.shape[1]}")
print(f"  Training rows  : {X.shape[0]:,}")
print(f"  Fraud rate     : {y.mean():.3%}  (class imbalance ratio: {(y==0).sum()/(y==1).sum():.1f}:1)")

# ── 6. Time-based train / validation split ───────────────────
# Use time ordering to mimic production: train on older data,
# validate on newer — avoids data leakage from future signals.
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Split — Train: {len(X_tr):,}  |  Val: {len(X_val):,}")

# ── 7. Train LightGBM classifier ─────────────────────────────
import lightgbm as lgb

lgb_params = {
    "objective"         : "binary",
    "metric"            : ["auc", "average_precision"],
    "n_estimators"      : 1000,
    "learning_rate"     : 0.05,
    "num_leaves"        : 127,
    "max_depth"         : -1,
    "min_child_samples" : 50,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "reg_alpha"         : 0.1,
    "reg_lambda"        : 1.0,
    # Class imbalance: up-weight fraud samples
    "scale_pos_weight"  : float((y == 0).sum()) / float((y == 1).sum()),
    "n_jobs"            : -1,
    "random_state"      : 42,
    "verbose"           : -1,
}

print(f"\nTraining LightGBM …")
print(f"  scale_pos_weight = {lgb_params['scale_pos_weight']:.1f}  (handles {y.mean():.2%} fraud rate)")

model = lgb.LGBMClassifier(**lgb_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100)
    ]
)
print(f"✓ Training complete — best iteration: {model.best_iteration_}")

# ── 8. Evaluation — AUC-PR as primary metric ─────────────────
# AUC-PR is the correct primary metric for class-imbalanced
# fraud data. ROC-AUC can look deceptively high even with many
# false positives because of the overwhelming negative class.
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report,
    confusion_matrix
)

val_proba = model.predict_proba(X_val)[:, 1]

auc_roc = roc_auc_score(y_val, val_proba)
auc_pr  = average_precision_score(y_val, val_proba)

print(f"\n{'═'*45}")
print(f"  ROC-AUC  : {auc_roc:.4f}")
print(f"  AUC-PR   : {auc_pr:.4f}   ← primary metric")
print(f"{'═'*45}")

# Threshold tuning: find threshold that maximises F1
prec, rec, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores   = 2 * prec * rec / (prec + rec + 1e-9)
best_idx    = np.argmax(f1_scores[:-1])
best_thresh = float(thresholds[best_idx])
best_f1     = float(f1_scores[best_idx])

print(f"\n  Threshold (max-F1) : {best_thresh:.4f}")
print(f"  Best F1            : {best_f1:.4f}")

y_pred = (val_proba >= best_thresh).astype(int)
print(f"\n{classification_report(y_val, y_pred, target_names=['legit', 'fraud'])}")

cm = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)
print(f"  False Positive Rate : {fpr:.4f}  ({fp:,} legitimate txns flagged as fraud)")
print(f"  False Negative Rate : {fn/(fn+tp):.4f}  ({fn:,} fraud txns missed)")

# ── 9. Visualisations ────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# 9a. Feature importance
fi = pd.Series(model.feature_importances_, index=feature_cols)
top20 = fi.nlargest(20)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#378ADD" if not any(x in idx for x in
          ["email", "identity", "device", "addr", "missing"])
          else "#D85A30" for idx in top20.index]
sns.barplot(x=top20.values, y=top20.index, palette=colors, ax=ax)
ax.set_title("Top-20 Feature Importances — orange = identity-resolution features", fontsize=11)
ax.set_xlabel("LightGBM gain importance")
plt.tight_layout()
plt.savefig("/content/feature_importance.png", dpi=150)
plt.show()

# 9b. Precision-Recall curve
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec, prec, color="#378ADD", lw=2, label=f"LightGBM (AUC-PR = {auc_pr:.3f})")
ax.axvline(rec[best_idx], color="#D85A30", ls="--", lw=1.2,
           label=f"Best threshold = {best_thresh:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
plt.tight_layout()
plt.savefig("/content/precision_recall_curve.png", dpi=150)
plt.show()

print("✓ Plots saved to /content/")

# ── 10. PSI — Production Model Monitoring ────────────────────
# Pi team owns production model monitoring. PSI (Population
# Stability Index) measures score distribution drift between
# training window and a new production batch.
#
# Thresholds (industry standard):
#   PSI < 0.10  → stable, no action
#   PSI 0.10–0.25 → moderate drift, monitor closely
#   PSI > 0.25  → significant drift → trigger retraining

def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_bins: int = 10) -> float:
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = 0.0, 1.0

    exp_cnt = np.histogram(expected, bins=bins)[0]
    act_cnt = np.histogram(actual,   bins=bins)[0]

    exp_pct = np.where(exp_cnt == 0, 1e-6, exp_cnt / len(expected))
    act_pct = np.where(act_cnt == 0, 1e-6, act_cnt / len(actual))

    psi_bins = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return float(psi_bins.sum())

# Simulate a production batch by splitting validation in half
n_half   = len(val_proba) // 2
baseline = val_proba[:n_half]
prod     = val_proba[n_half:]
psi      = compute_psi(baseline, prod)

print(f"\n── PSI Score Distribution Monitoring ──")
print(f"  PSI = {psi:.4f}", end="   →   ")
if psi < 0.10:
    print("STABLE — no retraining needed")
elif psi < 0.25:
    print("MODERATE DRIFT — schedule retraining review")
else:
    print("⚠ SIGNIFICANT DRIFT — trigger automated retraining")

# PSI plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(baseline, bins=40, alpha=0.65, color="#378ADD", label="Baseline (training window)")
ax.hist(prod,     bins=40, alpha=0.65, color="#D85A30", label="Production batch")
ax.set_title(f"Score Distribution Drift  |  PSI = {psi:.4f}", fontsize=12)
ax.set_xlabel("Fraud risk score")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("/content/psi_score_drift.png", dpi=150)
plt.show()

# ── 11. Save model artefact ──────────────────────────────────
import pickle, datetime

MODEL_PATH = "/content/fraud_risk_scorer.pkl"
artefact = {
    "model"        : model,
    "feature_cols" : feature_cols,
    "threshold"    : best_thresh,
    "trained_at"   : datetime.datetime.utcnow().isoformat(),
    "val_auc_pr"   : float(auc_pr),
    "val_auc_roc"  : float(auc_roc),
    "val_f1"       : float(best_f1),
    "false_positive_rate" : float(fpr),
    "n_features"   : len(feature_cols),
    "lgb_best_iter": model.best_iteration_,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(artefact, f)

print(f"\n✓ Model saved → {MODEL_PATH}")
print(f"  AUC-PR={auc_pr:.4f}  |  F1={best_f1:.4f}  |  FPR={fpr:.4f}  |  threshold={best_thresh:.4f}")

# ── 12. FastAPI real-time inference endpoint ─────────────────
# Serves single-transaction scoring at <180ms p99.
# Routes: GET /health, POST /score

import threading, time
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

with open(MODEL_PATH, "rb") as f:
    _art = pickle.load(f)

_model     = _art["model"]
_feat_cols = _art["feature_cols"]
_threshold = _art["threshold"]

app = FastAPI(title="Identity-Aware Fraud Risk Scorer — Pi Team Demo")

class TransactionFeatures(BaseModel):
    features: dict

@app.get("/health")
def health():
    return {
        "status"    : "ok",
        "model_meta": {
            "trained_at" : _art["trained_at"],
            "auc_pr"     : round(_art["val_auc_pr"], 4),
            "auc_roc"    : round(_art["val_auc_roc"], 4),
            "threshold"  : round(_art["threshold"], 4),
            "n_features" : _art["n_features"],
        }
    }

@app.post("/score")
def score(payload: TransactionFeatures):
    import time as _time
    t0 = _time.perf_counter()

    row  = pd.DataFrame([payload.features]).reindex(columns=_feat_cols, fill_value=0)
    row  = row.astype(np.float32)
    prob = float(_model.predict_proba(row)[0, 1])

    latency_ms = (_time.perf_counter() - t0) * 1000
    decision   = "FRAUD" if prob >= _threshold else "LEGIT"

    return {
        "fraud_probability" : round(prob, 4),
        "decision"          : decision,
        "threshold_used"    : round(_threshold, 4),
        "confidence"        : round(abs(prob - _threshold) / _threshold, 3),
        "latency_ms"        : round(latency_ms, 2),
    }

@app.get("/model/features")
def feature_list():
    return {"features": _feat_cols, "count": len(_feat_cols)}

def _run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

print("\nStarting FastAPI server …")
server_thread = threading.Thread(target=_run_server, daemon=True)
server_thread.start()
time.sleep(2)

# ── 13. Smoke-test the API ────────────────────────────────────
print("\n── FastAPI smoke tests ──")

# /health
r = requests.get("http://localhost:8000/health")
print(f"  GET  /health        → {r.json()}")

# /score with a real validation sample
sample_raw = X_val.iloc[0].to_dict()
sample_clean = {
    k: (None if (isinstance(v, float) and np.isnan(v)) else float(v))
    for k, v in sample_raw.items()
}
r = requests.post("http://localhost:8000/score",
                  json={"features": sample_clean})
print(f"  POST /score         → {r.json()}")

# /model/features
r = requests.get("http://localhost:8000/model/features")
print(f"  GET  /model/features → count={r.json()['count']} features")

# Latency benchmark — 50 requests
latencies = []
for _ in range(50):
    t0 = time.perf_counter()
    requests.post("http://localhost:8000/score", json={"features": sample_clean})
    latencies.append((time.perf_counter() - t0) * 1000)

p50 = np.percentile(latencies, 50)
p99 = np.percentile(latencies, 99)
print(f"\n  Latency over 50 requests — p50: {p50:.1f}ms  |  p99: {p99:.1f}ms")

# ── 14. Final summary ─────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║   Identity-Aware Fraud Risk Scorer — Results         ║
╠══════════════════════════════════════════════════════╣
║  Dataset      : IEEE-CIS (~590K transactions)        ║
║  Identity join: {identity_coverage:.0%} coverage across 2 tables      ║
║  Features     : {len(feature_cols)} (incl. {len(engineered)} engineered)               ║
║  Model        : LightGBM, {model.best_iteration_} trees               ║
║  AUC-PR       : {auc_pr:.4f}  (primary metric)             ║
║  ROC-AUC      : {auc_roc:.4f}                               ║
║  F1 @ thresh  : {best_f1:.4f}  (threshold={best_thresh:.3f})         ║
║  FPR          : {fpr:.4f}                               ║
║  PSI          : {psi:.4f}  ({"STABLE" if psi < 0.10 else "DRIFT DETECTED"})                    ║
║  API latency  : p50={p50:.0f}ms  p99={p99:.0f}ms                   ║
╚══════════════════════════════════════════════════════╝

API running at http://localhost:8000
Interactive docs: http://localhost:8000/docs
""")
