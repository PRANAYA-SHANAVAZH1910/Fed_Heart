import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def simulate_federated_hospitals(
    df=pd.read_csv("/Users/pranaya.shanavazh/Desktop/ED For Heart/preprocessed_train_data.csv"),
    random_state=42,
    temperature=0.9,
    balance_strength=0.35,
    save_plots=True,
    output_dir="hospital_simulation_outputs"
):
    """
    Simulate 10 non-IID hospitals from a preprocessed Framingham dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the required columns.
    random_state : int
        Random seed for reproducibility.
    temperature : float
        Softmax temperature. Lower = sharper assignments, higher = more overlap.
    balance_strength : float
        Strength of soft balancing toward equal hospital sizes.
    save_plots : bool
        Whether to save plots.
    output_dir : str
        Directory for saving plots and CSV files.

    Returns
    -------
    results : dict
        Dictionary containing:
        - patient_df
        - quartiles
        - hospital_stats
        - binary_stats
        - class_distribution
        - hospital_sizes
        - quartile_distribution
        - binary_distribution
        - feature_means
        - score_matrix
        - prob_matrix
        - profiles
    """

    rng = np.random.default_rng(random_state)
    work_df = df.copy()

    continuous_features = [
        "num__age",
        "num__cigsPerDay",
        "num__totChol",
        "num__sysBP",
        "num__diaBP",
        "num__BMI",
        "num__heartRate",
        "num__glucose"
    ]

    binary_features = [
        "num__male",
        "num__currentSmoker",
        "num__BPMeds",
        "num__prevalentStroke",
        "num__prevalentHyp",
        "num__diabetes"
    ]

    target_col = "TenYearCHD"

    target_col = "TenYearCHD"

    required = continuous_features + binary_features
    missing = [c for c in required if c not in work_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_target = target_col in work_df.columns

    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Compute quartiles for continuous features
    # --------------------------------------------------
    quartile_summary = {}
    quartile_midpoints = {}
    quartile_cols = []

    for col in continuous_features:
        q1, q2, q3 = work_df[col].quantile([0.25, 0.50, 0.75]).tolist()
        quartile_summary[col] = {"Q1": q1, "Q2": q2, "Q3": q3}

        qcol = f"{col}__quartile"
        labels = [1, 2, 3, 4]

        try:
            work_df[qcol] = pd.qcut(work_df[col], q=4, labels=labels, duplicates="drop").astype(int)
        except Exception:
            ranks = work_df[col].rank(method="first")
            work_df[qcol] = pd.qcut(ranks, q=4, labels=labels).astype(int)

        quartile_cols.append(qcol)

        grouped = work_df.groupby(qcol)[col].mean().reindex(labels)
        grouped = grouped.fillna(work_df[col].mean())
        quartile_midpoints[col] = grouped.to_dict()

    quartiles_df = pd.DataFrame(quartile_summary).T
    quartiles_df.index.name = "feature"

    # Base binary rates from data for neutral hospitals
    binary_base_rates = work_df[binary_features].mean().to_dict()

    # --------------------------------------------------
    # 2. Define 10 hospital profiles
    # Continuous: preferred quartiles
    # Binary: target probabilities
    # --------------------------------------------------
    hospital_profiles = {
        1: {
            "name": "Young healthy population",
            "continuous": {
                "num__age": 1,
                "num__cigsPerDay": 1,
                "num__totChol": 1,
                "num__sysBP": 1,
                "num__diaBP": 1,
                "num__BMI": 1,
                "num__heartRate": 2,
                "num__glucose": 1,
            },
            "binary": {
                "num__male": 0.45,
                "num__currentSmoker": 0.18,
                "num__BPMeds": 0.03,
                "num__prevalentStroke": 0.01,
                "num__prevalentHyp": 0.10,
                "num__diabetes": 0.04,
            },
        },
        2: {
            "name": "General community",
            "continuous": {
                "num__age": 2,
                "num__cigsPerDay": 2,
                "num__totChol": 2,
                "num__sysBP": 2,
                "num__diaBP": 2,
                "num__BMI": 2,
                "num__heartRate": 2,
                "num__glucose": 2,
            },
            "binary": {k: float(v) for k, v in binary_base_rates.items()},
        },
        3: {
            "name": "Hypertension clinic",
            "continuous": {
                "num__age": 3,
                "num__cigsPerDay": 2,
                "num__totChol": 3,
                "num__sysBP": 4,
                "num__diaBP": 4,
                "num__BMI": 3,
                "num__heartRate": 3,
                "num__glucose": 2,
            },
            "binary": {
                "num__male": 0.50,
                "num__currentSmoker": 0.28,
                "num__BPMeds": 0.35,
                "num__prevalentStroke": 0.05,
                "num__prevalentHyp": 0.85,
                "num__diabetes": 0.12,
            },
        },
        4: {
            "name": "Diabetes & obesity clinic",
            "continuous": {
                "num__age": 3,
                "num__cigsPerDay": 1,
                "num__totChol": 3,
                "num__sysBP": 3,
                "num__diaBP": 3,
                "num__BMI": 4,
                "num__heartRate": 3,
                "num__glucose": 4,
            },
            "binary": {
                "num__male": 0.45,
                "num__currentSmoker": 0.20,
                "num__BPMeds": 0.18,
                "num__prevalentStroke": 0.03,
                "num__prevalentHyp": 0.55,
                "num__diabetes": 0.75,
            },
        },
        5: {
            "name": "Heavy smokers",
            "continuous": {
                "num__age": 2,
                "num__cigsPerDay": 4,
                "num__totChol": 3,
                "num__sysBP": 3,
                "num__diaBP": 2,
                "num__BMI": 2,
                "num__heartRate": 3,
                "num__glucose": 2,
            },
            "binary": {
                "num__male": 0.65,
                "num__currentSmoker": 0.90,
                "num__BPMeds": 0.08,
                "num__prevalentStroke": 0.03,
                "num__prevalentHyp": 0.35,
                "num__diabetes": 0.10,
            },
        },
        6: {
            "name": "Cardiology referral center",
            "continuous": {
                "num__age": 4,
                "num__cigsPerDay": 3,
                "num__totChol": 4,
                "num__sysBP": 4,
                "num__diaBP": 3,
                "num__BMI": 3,
                "num__heartRate": 3,
                "num__glucose": 3,
            },
            "binary": {
                "num__male": 0.60,
                "num__currentSmoker": 0.38,
                "num__BPMeds": 0.28,
                "num__prevalentStroke": 0.10,
                "num__prevalentHyp": 0.78,
                "num__diabetes": 0.22,
            },
        },
        7: {
            "name": "Women's preventive clinic",
            "continuous": {
                "num__age": 2,
                "num__cigsPerDay": 1,
                "num__totChol": 2,
                "num__sysBP": 2,
                "num__diaBP": 2,
                "num__BMI": 2,
                "num__heartRate": 2,
                "num__glucose": 2,
            },
            "binary": {
                "num__male": 0.08,
                "num__currentSmoker": 0.16,
                "num__BPMeds": 0.06,
                "num__prevalentStroke": 0.01,
                "num__prevalentHyp": 0.18,
                "num__diabetes": 0.05,
            },
        },
        8: {
            "name": "Geriatric hospital",
            "continuous": {
                "num__age": 4,
                "num__cigsPerDay": 1,
                "num__totChol": 2,
                "num__sysBP": 4,
                "num__diaBP": 3,
                "num__BMI": 2,
                "num__heartRate": 2,
                "num__glucose": 3,
            },
            "binary": {
                "num__male": 0.42,
                "num__currentSmoker": 0.10,
                "num__BPMeds": 0.24,
                "num__prevalentStroke": 0.08,
                "num__prevalentHyp": 0.72,
                "num__diabetes": 0.18,
            },
        },
        9: {
            "name": "Metabolic syndrome clinic",
            "continuous": {
                "num__age": 3,
                "num__cigsPerDay": 2,
                "num__totChol": 4,
                "num__sysBP": 4,
                "num__diaBP": 4,
                "num__BMI": 4,
                "num__heartRate": 3,
                "num__glucose": 4,
            },
            "binary": {
                "num__male": 0.52,
                "num__currentSmoker": 0.30,
                "num__BPMeds": 0.22,
                "num__prevalentStroke": 0.04,
                "num__prevalentHyp": 0.76,
                "num__diabetes": 0.60,
            },
        },
        10: {
            "name": "General tertiary hospital",
            "continuous": {
                "num__age": 3,
                "num__cigsPerDay": 2,
                "num__totChol": 3,
                "num__sysBP": 3,
                "num__diaBP": 3,
                "num__BMI": 3,
                "num__heartRate": 3,
                "num__glucose": 3,
            },
            "binary": {
                "num__male": 0.50,
                "num__currentSmoker": 0.26,
                "num__BPMeds": 0.14,
                "num__prevalentStroke": 0.03,
                "num__prevalentHyp": 0.42,
                "num__diabetes": 0.14,
            },
        },
    }

    # --------------------------------------------------
    # 3. Similarity scoring + softmax + random assignment
    #    with approximate balancing
    # --------------------------------------------------
    hospital_ids = list(hospital_profiles.keys())
    n_hospitals = len(hospital_ids)
    n = len(work_df)
    target_size = n / n_hospitals

    assigned_counts = {hid: 0 for hid in hospital_ids}
    assignments = []

    score_matrix = np.zeros((n, n_hospitals))
    prob_matrix = np.zeros((n, n_hospitals))

    cont_weight = 1.0
    bin_weight = 1.2
    local_value_bonus = 0.35

    shuffled_idx = rng.permutation(work_df.index.to_numpy())

    for idx in shuffled_idx:
        row = work_df.loc[idx]
        scores = []

        for hid in hospital_ids:
            profile = hospital_profiles[hid]
            score = 0.0

            # Continuous features: quartile alignment + distance-to-quartile-center bonus
            for col in continuous_features:
                patient_q = int(row[f"{col}__quartile"])
                target_q = profile["continuous"][col]

                quartile_match_score = 3 - abs(patient_q - target_q)
                score += cont_weight * quartile_match_score

                patient_val = float(row[col])
                target_val = quartile_midpoints[col][target_q]
                scale = float(work_df[col].std(ddof=0)) + 1e-8
                proximity_bonus = np.exp(-abs(patient_val - target_val) / scale)
                score += local_value_bonus * proximity_bonus

            # Binary features: probability-based matching
            for col in binary_features:
                patient_val = float(row[col])
                target_p = float(profile["binary"][col])

                # probability of observing this binary value under hospital's target rate
                match_prob = patient_val * target_p + (1 - patient_val) * (1 - target_p)
                score += bin_weight * match_prob

            # Soft balancing penalty for hospitals already above target size
            overload = max(0.0, (assigned_counts[hid] - target_size) / max(target_size, 1))
            score -= balance_strength * overload

            scores.append(score)

        scores = np.array(scores, dtype=float)

        # Softmax
        stabilized = (scores - scores.max()) / temperature
        probs = np.exp(stabilized)
        probs = probs / probs.sum()

        chosen_hospital = int(rng.choice(hospital_ids, p=probs))
        assignments.append((idx, chosen_hospital))
        assigned_counts[chosen_hospital] += 1

        row_pos = work_df.index.get_loc(idx)
        score_matrix[row_pos, :] = scores
        prob_matrix[row_pos, :] = probs

    assignment_df = pd.DataFrame(assignments, columns=["index", "hospital_id"]).set_index("index")
    work_df = work_df.join(assignment_df, how="left")
    work_df["hospital_id"] = work_df["hospital_id"].astype(int)
    work_df["hospital_name"] = work_df["hospital_id"].map(
        {hid: hospital_profiles[hid]["name"] for hid in hospital_ids}
    )

    # --------------------------------------------------
    # 4. Statistics and distributions
    # --------------------------------------------------
    hospital_sizes = (
        work_df.groupby(["hospital_id", "hospital_name"])
        .size()
        .reset_index(name="n_patients")
        .sort_values("hospital_id")
    )

    hospital_stats = (
        work_df.groupby(["hospital_id", "hospital_name"])[continuous_features]
        .agg(["mean", "std", "median"])
    )
    hospital_stats.columns = [f"{c[0]}__{c[1]}" for c in hospital_stats.columns]
    hospital_stats = hospital_stats.reset_index()

    binary_stats = (
        work_df.groupby(["hospital_id", "hospital_name"])[binary_features]
        .mean()
        .reset_index()
    )
    binary_stats = binary_stats.rename(columns={c: f"{c}__rate" for c in binary_features})

    if has_target:
        class_distribution = (
        work_df.groupby(["hospital_id", "hospital_name"])[target_col]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={
            "mean": "chd_prevalence",
            "sum": "chd_cases",
            "count": "total_cases"
        })
    )
    else:
        class_distribution = None

    quartile_distribution_list = []
    for col in continuous_features:
        qcol = f"{col}__quartile"
        temp = (
            work_df.groupby(["hospital_id", qcol])
            .size()
            .reset_index(name="count")
            .rename(columns={qcol: "quartile"})
        )
        temp["feature"] = col
        quartile_distribution_list.append(temp)
    quartile_distribution = pd.concat(quartile_distribution_list, ignore_index=True)

    binary_distribution_list = []
    for col in binary_features:
        temp = (
            work_df.groupby("hospital_id")[col]
            .value_counts(dropna=False)
            .reset_index(name="count")
            .rename(columns={col: "value"})
        )
        temp["feature"] = col
        binary_distribution_list.append(temp)
    binary_distribution = pd.concat(binary_distribution_list, ignore_index=True)

    feature_means = (
        work_df.groupby(["hospital_id", "hospital_name"])[continuous_features + binary_features]
        .mean()
        .reset_index()
    )

    score_df = pd.DataFrame(
        score_matrix,
        index=work_df.index,
        columns=[f"hospital_{hid}_score" for hid in hospital_ids]
    )

    prob_df = pd.DataFrame(
        prob_matrix,
        index=work_df.index,
        columns=[f"hospital_{hid}_prob" for hid in hospital_ids]
    )

    # --------------------------------------------------
    # 5. Save tables
    # --------------------------------------------------
    if save_plots:
        quartiles_df.to_csv(f"{output_dir}/quartiles.csv")
        hospital_sizes.to_csv(f"{output_dir}/hospital_sizes.csv", index=False)
        hospital_stats.to_csv(f"{output_dir}/hospital_statistics.csv", index=False)
        binary_stats.to_csv(f"{output_dir}/binary_feature_rates.csv", index=False)
        if class_distribution is not None:
            class_distribution.to_csv(f"{output_dir}/class_distribution.csv", index=False)
        quartile_distribution.to_csv(f"{output_dir}/quartile_distribution.csv", index=False)
        binary_distribution.to_csv(f"{output_dir}/binary_distribution.csv", index=False)
        feature_means.to_csv(f"{output_dir}/hospital_feature_means.csv", index=False)
        work_df.to_csv(f"{output_dir}/patients_with_hospital_id.csv", index=False)

    

from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    csv_path = Path("/Users/pranaya.shanavazh/Desktop/ED For Heart/preprocessed_train_data.csv")
    df = pd.read_csv(csv_path)

    results = simulate_federated_hospitals(df)
    final_df = results["patient_df"]
    print(final_df.head())

    final_df = results["patient_df"]

    print("\nFirst 5 rows:")
    print(final_df.head())

    print("\nHospital sizes:")
    print(results["hospital_sizes"])

    if results["class_distribution"] is not None:
        print("\nClass distribution:")
        print(results["class_distribution"])
    else:
        print("\nClass distribution skipped because TenYearCHD is not present.")

    print("\nQuartiles:")
    print(results["quartiles"])

    final_df.to_csv("final_patients_with_hospital_id.csv", index=False)
    print("\nSaved final_patients_with_hospital_id.csv")
# --------------------------------------------------
# Example usage
# --------------------------------------------------
# Suppose your preprocessed dataframe is called df
#
# results = simulate_federated_hospitals(
#     df,
#     random_state=42,
#     temperature=0.9,
#     balance_strength=0.35,
#     save_plots=True,
#     output_dir="hospital_simulation_outputs"
# )
#
# final_df = results["patient_df"]
# print(final_df.head())
# print(results["hospital_sizes"])
# print(results["class_distribution"])
# print(results["quartiles"])