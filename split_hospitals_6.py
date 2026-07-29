import os
import pandas as pd
import numpy as np

def create_6_hospital_split(input_csv: str = "./preprocessed_augmented.csv", output_dir: str = "./Gemini_split_new"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    print(f"📥 Loaded dataset with {len(df)} records from '{input_csv}'")

    # Pre-calculate quartiles for continuous features
    q = {
        'age_q25': df['age'].quantile(0.25), 'age_q75': df['age'].quantile(0.75),
        'sysBP_q50': df['sysBP'].quantile(0.50), 'sysBP_q75': df['sysBP'].quantile(0.75),
        'glucose_q50': df['glucose'].quantile(0.50), 'glucose_q75': df['glucose'].quantile(0.75),
        'BMI_q50': df['BMI'].quantile(0.50),
    }

    n_samples = len(df) // 6  # Rough samples per hospital (~850 records each)

    # Define selection filters / weights per hospital profile
    profiles = {
        1: ("H1_Healthy_Young", (df['age'] <= q['age_q25']) & (df['sysBP'] <= q['sysBP_q50'])),
        2: ("H2_Hypertension_Cardiac", (df['sysBP'] >= q['sysBP_q75']) | (df['prevalentHyp'] == 1)),
        3: ("H3_Diabetic_Metabolic", (df['glucose'] >= q['glucose_q75']) | (df['diabetes'] == 1) | (df['BMI'] >= q['BMI_q50'])),
        4: ("H4_Smokers_Vascular", (df['currentSmoker'] == 1) | (df['prevalentStroke'] == 1)),
        5: ("H5_Geriatric_Elderly", (df['age'] >= q['age_q75'])),
        6: ("H6_Mixed_Community", None)  # Uniform sample fallback
    }

    used_indices = set()

    for h_id in range(1, 7):
        name, condition = profiles[h_id]
        available_df = df[~df.index.isin(used_indices)]

        if condition is not None:
            # Filter rows matching clinical profile
            matched_indices = available_df[condition].index
            if len(matched_indices) >= n_samples:
                selected_indices = np.random.choice(matched_indices, size=n_samples, replace=False)
            else:
                # If matched rows < desired batch size, fill remainder from general pool
                remaining_size = n_samples - len(matched_indices)
                other_indices = available_df[~available_df.index.isin(matched_indices)].index
                fill_indices = np.random.choice(other_indices, size=remaining_size, replace=False)
                selected_indices = np.concatenate([matched_indices, fill_indices])
        else:
            # General mixed hospital
            selected_indices = np.random.choice(available_df.index, size=min(n_samples, len(available_df)), replace=False)

        used_indices.update(selected_indices)
        hospital_df = df.loc[selected_indices].sample(frac=1).reset_index(drop=True)  # Shuffle local client data
        
        output_file = os.path.join(output_dir, f"Hospital_{h_id}_data.csv")
        hospital_df.to_csv(output_file, index=False)

        pos_rate = (hospital_df['TenYearCHD'] == 1).mean() * 100
        print(f"✅ Created {name} -> 'Hospital_{h_id}_data.csv' | Rows: {len(hospital_df)} | Pos CHD Rate: {pos_rate:.1f}%")

    print(f"\n🎉 Successfully created all 6 hospital CSV splits in '{output_dir}/'")

if __name__ == "__main__":
    create_6_hospital_split()