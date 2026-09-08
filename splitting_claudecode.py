"""
Federated Learning Hospital Simulator for the Framingham Heart Study dataset.

Splits a single preprocessed dataframe into 10 non-IID "hospital" subsets by:
  1. Building a probabilistic profile for each hospital over the continuous
     features (quartile preference weights) and binary features (target
     prevalence).
  2. Scoring every patient's log-likelihood under each hospital profile
     (a naive-Bayes-style similarity score).
  3. Converting scores to per-patient assignment probabilities with a
     temperature-scaled softmax (so hospitals overlap - this is NOT a hard
     rule-based split).
  4. Sampling each patient's hospital from those probabilities, with a
     streaming size-balancing correction so all 10 hospitals end up
     approximately equal in size while still respecting each patient's
     softmax preferences.

Works whether the input columns are raw units or already scaled (e.g. via a
sklearn `num__` ColumnTransformer/StandardScaler): quartiles are computed
directly from whatever distribution is present, and binary columns are
detected by their two unique values rather than assuming literal 0/1.

Usage:
    from federated_hospital_partition import run_pipeline
    result = run_pipeline(df, output_dir="/mnt/user-data/outputs", seed=42)
    result["dataframe"]          # full dataframe with hospital_id column
    result["hospital_sizes"]     # patients per hospital
    result["hospital_stats"]     # per-hospital feature means / CHD prevalence
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from splitting_claudecode import run_pipeline

df = pd.read_csv("'/Users/pranaya.shanavazh/Desktop/ED For Heart/preprocessed_train_data.csv'")
result = run_pipeline(df)
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# 1. Column configuration
# --------------------------------------------------------------------------

CONTINUOUS_FEATURES = [
    "num__age", "num__cigsPerDay", "num__totChol", "num__sysBP",
    "num__diaBP", "num__BMI", "num__heartRate", "num__glucose",
]

BINARY_FEATURES = [
    "num__male", "num__currentSmoker", "num__BPMeds",
    "num__prevalentStroke", "num__prevalentHyp", "num__diabetes",
]

LABEL_COL = "TenYearCHD"

HOSPITAL_NAMES = {
    1: "Young Healthy Population",
    2: "General Community",
    3: "Hypertension Clinic",
    4: "Diabetes & Obesity Clinic",
    5: "Heavy Smokers Clinic",
    6: "Cardiology Referral Center",
    7: "Women's Preventive Clinic",
    8: "Geriatric Hospital",
    9: "Metabolic Syndrome Clinic",
    10: "General Tertiary Hospital",
}

# --------------------------------------------------------------------------
# 2. Hospital profiles
#    - continuous: preference weights over (Q1, Q2, Q3, Q4), must sum to 1
#    - binary: target prevalence of the "positive" (higher-valued) class
#    A flat [0.25, 0.25, 0.25, 0.25] / 0.5 profile means "no particular
#    preference" (i.e. behaves like the general population for that feature).
# --------------------------------------------------------------------------

FLAT4 = [0.25, 0.25, 0.25, 0.25]

HOSPITAL_PROFILES = {
    1: {  # Young Healthy Population
        "continuous": {
            "num__age": [0.55, 0.30, 0.10, 0.05],
            "num__cigsPerDay": [0.55, 0.25, 0.12, 0.08],
            "num__totChol": [0.40, 0.30, 0.20, 0.10],
            "num__sysBP": [0.50, 0.30, 0.13, 0.07],
            "num__diaBP": [0.50, 0.30, 0.13, 0.07],
            "num__BMI": [0.45, 0.30, 0.15, 0.10],
            "num__heartRate": FLAT4,
            "num__glucose": [0.45, 0.30, 0.15, 0.10],
        },
        "binary": {
            "num__male": 0.50, "num__currentSmoker": 0.20, "num__BPMeds": 0.02,
            "num__prevalentStroke": 0.01, "num__prevalentHyp": 0.08, "num__diabetes": 0.02,
        },
    },
    2: {  # General Community
        "continuous": {c: FLAT4 for c in CONTINUOUS_FEATURES},
        "binary": {
            "num__male": 0.45, "num__currentSmoker": 0.45, "num__BPMeds": 0.10,
            "num__prevalentStroke": 0.02, "num__prevalentHyp": 0.30, "num__diabetes": 0.06,
        },
    },
    3: {  # Hypertension Clinic
        "continuous": {
            "num__age": [0.10, 0.20, 0.35, 0.35],
            "num__cigsPerDay": FLAT4,
            "num__totChol": [0.10, 0.20, 0.30, 0.40],
            "num__sysBP": [0.03, 0.10, 0.32, 0.55],
            "num__diaBP": [0.05, 0.12, 0.33, 0.50],
            "num__BMI": [0.10, 0.25, 0.35, 0.30],
            "num__heartRate": FLAT4,
            "num__glucose": [0.15, 0.25, 0.30, 0.30],
        },
        "binary": {
            "num__male": 0.48, "num__currentSmoker": 0.30, "num__BPMeds": 0.55,
            "num__prevalentStroke": 0.05, "num__prevalentHyp": 0.90, "num__diabetes": 0.15,
        },
    },
    4: {  # Diabetes & Obesity Clinic
        "continuous": {
            "num__age": [0.15, 0.30, 0.30, 0.25],
            "num__cigsPerDay": FLAT4,
            "num__totChol": [0.10, 0.25, 0.30, 0.35],
            "num__sysBP": [0.10, 0.25, 0.33, 0.32],
            "num__diaBP": [0.10, 0.25, 0.33, 0.32],
            "num__BMI": [0.03, 0.10, 0.32, 0.55],
            "num__heartRate": FLAT4,
            "num__glucose": [0.03, 0.10, 0.27, 0.60],
        },
        "binary": {
            "num__male": 0.45, "num__currentSmoker": 0.25, "num__BPMeds": 0.25,
            "num__prevalentStroke": 0.03, "num__prevalentHyp": 0.45, "num__diabetes": 0.70,
        },
    },
    5: {  # Heavy Smokers Clinic
        "continuous": {
            "num__age": [0.20, 0.35, 0.30, 0.15],
            "num__cigsPerDay": [0.02, 0.08, 0.30, 0.60],
            "num__totChol": [0.15, 0.25, 0.30, 0.30],
            "num__sysBP": [0.20, 0.30, 0.30, 0.20],
            "num__diaBP": [0.20, 0.30, 0.30, 0.20],
            "num__BMI": FLAT4,
            "num__heartRate": [0.15, 0.25, 0.30, 0.30],
            "num__glucose": FLAT4,
        },
        "binary": {
            "num__male": 0.60, "num__currentSmoker": 0.95, "num__BPMeds": 0.08,
            "num__prevalentStroke": 0.03, "num__prevalentHyp": 0.25, "num__diabetes": 0.06,
        },
    },
    6: {  # Cardiology Referral Center
        "continuous": {
            "num__age": [0.05, 0.15, 0.35, 0.45],
            "num__cigsPerDay": [0.15, 0.25, 0.30, 0.30],
            "num__totChol": [0.05, 0.15, 0.35, 0.45],
            "num__sysBP": [0.03, 0.12, 0.35, 0.50],
            "num__diaBP": [0.05, 0.15, 0.35, 0.45],
            "num__BMI": [0.10, 0.25, 0.33, 0.32],
            "num__heartRate": [0.10, 0.25, 0.33, 0.32],
            "num__glucose": [0.10, 0.25, 0.30, 0.35],
        },
        "binary": {
            "num__male": 0.58, "num__currentSmoker": 0.35, "num__BPMeds": 0.50,
            "num__prevalentStroke": 0.15, "num__prevalentHyp": 0.65, "num__diabetes": 0.20,
        },
    },
    7: {  # Women's Preventive Clinic
        "continuous": {
            "num__age": [0.30, 0.35, 0.20, 0.15],
            "num__cigsPerDay": [0.55, 0.25, 0.12, 0.08],
            "num__totChol": [0.25, 0.30, 0.25, 0.20],
            "num__sysBP": [0.35, 0.30, 0.20, 0.15],
            "num__diaBP": [0.35, 0.30, 0.20, 0.15],
            "num__BMI": [0.30, 0.30, 0.22, 0.18],
            "num__heartRate": FLAT4,
            "num__glucose": [0.35, 0.30, 0.20, 0.15],
        },
        "binary": {
            "num__male": 0.02, "num__currentSmoker": 0.20, "num__BPMeds": 0.06,
            "num__prevalentStroke": 0.01, "num__prevalentHyp": 0.15, "num__diabetes": 0.03,
        },
    },
    8: {  # Geriatric Hospital
        "continuous": {
            "num__age": [0.02, 0.08, 0.25, 0.65],
            "num__cigsPerDay": [0.55, 0.25, 0.12, 0.08],
            "num__totChol": [0.15, 0.25, 0.30, 0.30],
            "num__sysBP": [0.08, 0.20, 0.32, 0.40],
            "num__diaBP": [0.15, 0.30, 0.30, 0.25],
            "num__BMI": FLAT4,
            "num__heartRate": FLAT4,
            "num__glucose": [0.15, 0.25, 0.30, 0.30],
        },
        "binary": {
            "num__male": 0.42, "num__currentSmoker": 0.12, "num__BPMeds": 0.45,
            "num__prevalentStroke": 0.10, "num__prevalentHyp": 0.60, "num__diabetes": 0.18,
        },
    },
    9: {  # Metabolic Syndrome Clinic
        "continuous": {
            "num__age": [0.10, 0.25, 0.35, 0.30],
            "num__cigsPerDay": FLAT4,
            "num__totChol": [0.05, 0.15, 0.35, 0.45],
            "num__sysBP": [0.08, 0.20, 0.35, 0.37],
            "num__diaBP": [0.08, 0.20, 0.35, 0.37],
            "num__BMI": [0.03, 0.12, 0.35, 0.50],
            "num__heartRate": FLAT4,
            "num__glucose": [0.05, 0.15, 0.35, 0.45],
        },
        "binary": {
            "num__male": 0.48, "num__currentSmoker": 0.30, "num__BPMeds": 0.30,
            "num__prevalentStroke": 0.04, "num__prevalentHyp": 0.55, "num__diabetes": 0.45,
        },
    },
    10: {  # General Tertiary Hospital
        "continuous": {
            "num__age": [0.15, 0.25, 0.30, 0.30],
            "num__cigsPerDay": [0.30, 0.25, 0.25, 0.20],
            "num__totChol": [0.15, 0.25, 0.30, 0.30],
            "num__sysBP": [0.15, 0.25, 0.30, 0.30],
            "num__diaBP": [0.15, 0.25, 0.30, 0.30],
            "num__BMI": [0.15, 0.25, 0.30, 0.30],
            "num__heartRate": FLAT4,
            "num__glucose": [0.15, 0.25, 0.30, 0.30],
        },
        "binary": {
            "num__male": 0.48, "num__currentSmoker": 0.40, "num__BPMeds": 0.15,
            "num__prevalentStroke": 0.04, "num__prevalentHyp": 0.35, "num__diabetes": 0.10,
        },
    },
}

for hid, profile in HOSPITAL_PROFILES.items():
    for feat, weights in profile["continuous"].items():
        assert abs(sum(weights) - 1.0) < 1e-6, f"Hospital {hid} {feat} weights must sum to 1"

# --------------------------------------------------------------------------
# 3. Quartile binning
# --------------------------------------------------------------------------

def compute_quartile_edges(df, continuous_cols):
    """Return {col: array of 5 bin edges (Q0..Q4)} computed from the data."""
    edges = {}
    for col in continuous_cols:
        qs = df[col].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
        qs = np.unique(qs)
        if len(qs) < 5:
            # Degenerate/low-cardinality column: pad with tiny jitter so
            # pd.cut still produces 4 usable bins.
            lo, hi = df[col].min(), df[col].max()
            qs = np.linspace(lo, hi, 5)
        edges[col] = qs
    return edges


def assign_quartile_bins(df, continuous_cols, edges):
    """Return a DataFrame of the same shape with values in {0,1,2,3} per feature."""
    bins = pd.DataFrame(index=df.index)
    for col in continuous_cols:
        e = edges[col].copy()
        e[0] -= 1e-9
        e[-1] += 1e-9
        bins[col] = pd.cut(df[col], bins=e, labels=[0, 1, 2, 3]).astype(int)
    return bins


def detect_binary_positive_value(df, binary_cols):
    """Return {col: value_treated_as_positive} — the higher of the two unique
    values present, so this works whether columns are raw 0/1 or scaled."""
    positive_value = {}
    for col in binary_cols:
        uniques = sorted(df[col].dropna().unique())
        if len(uniques) == 1:
            positive_value[col] = uniques[0]
        else:
            positive_value[col] = uniques[-1]
    return positive_value


# --------------------------------------------------------------------------
# 4. Similarity scoring + softmax assignment
# --------------------------------------------------------------------------

def compute_log_likelihoods(df, quartile_bins, positive_value, profiles,
                             continuous_cols, binary_cols, eps=1e-3):
    """Return an (n_patients, n_hospitals) array of naive-Bayes-style
    log-likelihood scores of each patient under each hospital profile."""
    n = len(df)
    hospital_ids = sorted(profiles.keys())
    scores = np.zeros((n, len(hospital_ids)))

    is_positive = {col: (df[col].values == positive_value[col]).astype(float)
                   for col in binary_cols}

    for h_idx, hid in enumerate(hospital_ids):
        profile = profiles[hid]
        col_score = np.zeros(n)

        for col in continuous_cols:
            weights = np.array(profile["continuous"][col])
            weights = np.clip(weights, eps, None)
            bin_idx = quartile_bins[col].values
            col_score += np.log(weights[bin_idx])

        for col in binary_cols:
            p = np.clip(profile["binary"][col], eps, 1 - eps)
            pos = is_positive[col]
            col_score += pos * np.log(p) + (1 - pos) * np.log(1 - p)

        scores[:, h_idx] = col_score

    return scores, hospital_ids


def softmax_rows(scores, temperature=1.0):
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def balanced_probabilistic_assignment(probs, n_hospitals, seed=42,
                                       balance_strength=1.0):
    """
    Randomly assign each patient to a hospital according to `probs`
    (n_patients x n_hospitals), while nudging assignments so hospital sizes
    end up approximately equal.

    Implementation: process patients in random order; maintain running
    counts per hospital; before sampling, reweight each patient's
    probability vector by how *under-filled* each hospital currently is
    relative to its fair-share target. This keeps assignment stochastic
    (no deterministic rules) while preventing any hospital from running
    far ahead of or behind its target share.
    """
    rng = np.random.default_rng(seed)
    n_patients = probs.shape[0]
    order = rng.permutation(n_patients)

    target_per_hospital = n_patients / n_hospitals
    counts = np.zeros(n_hospitals)
    assignments = np.empty(n_patients, dtype=int)

    for i in order:
        p = probs[i].copy()
        # Under-filled hospitals get a boost, over-filled ones get suppressed.
        fill_ratio = counts / max(target_per_hospital, 1e-9)
        balance_factor = np.clip(1.0 - balance_strength * (fill_ratio - 1.0), 0.05, None)
        p = p * balance_factor
        p = p / p.sum()

        choice = rng.choice(n_hospitals, p=p)
        assignments[i] = choice
        counts[choice] += 1

    return assignments


# --------------------------------------------------------------------------
# 5. Orchestration
# --------------------------------------------------------------------------

def assign_hospitals(df, temperature=1.0, balance_strength=1.0, seed=42):
    """Full pipeline: returns df with an added `hospital_id` column (1-10)
    plus the per-patient assignment-probability matrix (for inspection)."""
    df = df.copy()

    edges = compute_quartile_edges(df, CONTINUOUS_FEATURES)
    quartile_bins = assign_quartile_bins(df, CONTINUOUS_FEATURES, edges)
    positive_value = detect_binary_positive_value(df, BINARY_FEATURES)

    scores, hospital_ids = compute_log_likelihoods(
        df, quartile_bins, positive_value, HOSPITAL_PROFILES,
        CONTINUOUS_FEATURES, BINARY_FEATURES,
    )
    probs = softmax_rows(scores, temperature=temperature)

    assignments_idx = balanced_probabilistic_assignment(
        probs, n_hospitals=len(hospital_ids), seed=seed,
        balance_strength=balance_strength,
    )
    df["hospital_id"] = [hospital_ids[i] for i in assignments_idx]

    prob_df = pd.DataFrame(probs, columns=[f"p_hospital_{h}" for h in hospital_ids],
                            index=df.index)

    return df, prob_df, edges, positive_value


# --------------------------------------------------------------------------
# 6. Statistics
# --------------------------------------------------------------------------

def hospital_sizes(df):
    return (df["hospital_id"].value_counts()
            .reindex(range(1, 11), fill_value=0)
            .rename_axis("hospital_id")
            .reset_index(name="n_patients")
            .assign(hospital_name=lambda d: d["hospital_id"].map(HOSPITAL_NAMES)))


def hospital_statistics(df):
    feature_cols = CONTINUOUS_FEATURES + BINARY_FEATURES
    stats = df.groupby("hospital_id")[feature_cols].mean()
    stats["n_patients"] = df.groupby("hospital_id").size()
    if LABEL_COL in df.columns:
        stats["CHD_prevalence"] = df.groupby("hospital_id")[LABEL_COL].mean()
    stats["hospital_name"] = stats.index.map(HOSPITAL_NAMES)
    return stats.reset_index()


def class_distribution(df):
    if LABEL_COL not in df.columns:
        return None
    ct = pd.crosstab(df["hospital_id"], df[LABEL_COL], normalize="index")
    ct.columns = [f"{LABEL_COL}={c}" for c in ct.columns]
    ct["hospital_name"] = ct.index.map(HOSPITAL_NAMES)
    return ct.reset_index()


# --------------------------------------------------------------------------
# 7. Visualizations
# --------------------------------------------------------------------------

def make_visualizations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    order = list(range(1, 11))
    labels = [f"H{h}" for h in order]
    paths = {}

    # 1. Histogram of hospital sizes
    fig, ax = plt.subplots(figsize=(9, 5))
    sizes = df["hospital_id"].value_counts().reindex(order, fill_value=0)
    ax.bar(labels, sizes.values, color=sns.color_palette("viridis", 10))
    ax.axhline(len(df) / 10, color="red", linestyle="--", label="Equal-share target")
    ax.set_title("Patients per Hospital")
    ax.set_ylabel("Number of patients")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(output_dir, "01_hospital_sizes.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["hospital_sizes"] = p

    # 2. CHD prevalence per hospital
    if LABEL_COL in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        prev = df.groupby("hospital_id")[LABEL_COL].mean().reindex(order)
        ax.bar(labels, prev.values, color=sns.color_palette("rocket", 10))
        ax.axhline(df[LABEL_COL].mean(), color="blue", linestyle="--", label="Overall prevalence")
        ax.set_title("Ten-Year CHD Prevalence per Hospital")
        ax.set_ylabel("CHD prevalence")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(output_dir, "02_chd_prevalence.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        paths["chd_prevalence"] = p

    # 3. Age distribution by hospital
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="hospital_id", y="num__age", order=order, hue="hospital_id",
                legend=False, ax=ax, palette="viridis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_title("Age Distribution by Hospital")
    fig.tight_layout()
    p = os.path.join(output_dir, "03_age_distribution.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["age_distribution"] = p

    # 4. BMI distribution by hospital
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="hospital_id", y="num__BMI", order=order, hue="hospital_id",
                legend=False, ax=ax, palette="mako")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_title("BMI Distribution by Hospital")
    fig.tight_layout()
    p = os.path.join(output_dir, "04_bmi_distribution.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["bmi_distribution"] = p

    # 5. Blood pressure distribution by hospital (sysBP + diaBP)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(data=df, x="hospital_id", y="num__sysBP", order=order, hue="hospital_id",
                legend=False, ax=axes[0], palette="flare")
    axes[0].set_xticks(range(len(labels))); axes[0].set_xticklabels(labels)
    axes[0].set_title("Systolic BP by Hospital")
    sns.boxplot(data=df, x="hospital_id", y="num__diaBP", order=order, hue="hospital_id",
                legend=False, ax=axes[1], palette="crest")
    axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels)
    axes[1].set_title("Diastolic BP by Hospital")
    fig.tight_layout()
    p = os.path.join(output_dir, "05_bp_distribution.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["bp_distribution"] = p

    # 6. Heatmap of hospital feature means (z-scored across hospitals for comparability)
    feature_cols = CONTINUOUS_FEATURES + BINARY_FEATURES
    means = df.groupby("hospital_id")[feature_cols].mean().reindex(order)
    means_z = (means - means.mean()) / means.std(ddof=0)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(means_z.T, annot=means.T.round(2), fmt="", cmap="coolwarm",
                center=0, ax=ax, cbar_kws={"label": "z-score across hospitals"})
    ax.set_xticklabels(labels)
    ax.set_title("Hospital Feature Means (color = relative level, numbers = raw mean)")
    fig.tight_layout()
    p = os.path.join(output_dir, "06_feature_heatmap.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    paths["feature_heatmap"] = p

    return paths


# --------------------------------------------------------------------------
# 8. Top-level runner
# --------------------------------------------------------------------------

def run_pipeline(df, output_dir="/mnt/user-data/outputs", temperature=1.0,
                  balance_strength=20.0, seed=42, save_csv=True):
    missing = [c for c in CONTINUOUS_FEATURES + BINARY_FEATURES + [LABEL_COL]
               if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    result_df, prob_df, edges, positive_value = assign_hospitals(
        df, temperature=temperature, balance_strength=balance_strength, seed=seed,
    )

    sizes = hospital_sizes(result_df)
    stats = hospital_statistics(result_df)
    class_dist = class_distribution(result_df)
    plot_paths = make_visualizations(result_df, output_dir)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(os.path.join(output_dir, "patients_with_hospital_id.csv"), index=False)
        stats.to_csv(os.path.join(output_dir, "hospital_statistics.csv"), index=False)
        sizes.to_csv(os.path.join(output_dir, "hospital_sizes.csv"), index=False)
        if class_dist is not None:
            class_dist.to_csv(os.path.join(output_dir, "hospital_class_distribution.csv"), index=False)

    return {
        "dataframe": result_df,
        "assignment_probabilities": prob_df,
        "hospital_sizes": sizes,
        "hospital_stats": stats,
        "class_distribution": class_dist,
        "plot_paths": plot_paths,
    }


if __name__ == "__main__":
    print("Import this module and call run_pipeline(df) with your preprocessed "
          "Framingham DataFrame.")