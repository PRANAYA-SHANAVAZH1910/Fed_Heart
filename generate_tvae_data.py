import os
import pandas as pd
from ctgan import TVAE

def generate_synthetic_data(
    input_csv: str = "./framingham_clean_dropping.csv", 
    output_csv: str = "./preprocessed_augmented.csv", 
    num_synthetic_rows: int = 1500
):
    print("==========================================")
    print("🧬 TVAE SYNTHETIC DATA GENERATION PHASE")
    print("==========================================")
    
    # 1. Load Preprocessed Clean Dataset
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"❌ Input file '{input_csv}' not found. Please check your path.")
        
    df_real = pd.read_csv(input_csv)
    print(f"📥 Real Dataset Loaded: {len(df_real)} patient records")

    # 2. Define Discrete / Categorical Columns (Framingham / Heart Disease Features)
    discrete_columns = [
        'male', 'education', 'currentSmoker', 'BPMeds', 
        'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD'
    ]
    # Filter to only keep columns that actually exist in your dataset
    discrete_columns = [col for col in discrete_columns if col in df_real.columns]

    # 3. Initialize & Fit TVAE Model
    print("⏳ Training TVAE on complete preprocessed dataset (Epochs=300)...")
    tvae = TVAE(epochs=300, batch_size=64)
    tvae.fit(df_real, discrete_columns)
    print("✅ TVAE Training Complete!")

    # 4. Generate Synthetic Data
    print(f"🎲 Generating {num_synthetic_rows} synthetic patient records...")
    df_synthetic = tvae.sample(num_synthetic_rows)

    # 5. Concatenate Real + Synthetic Records
    df_augmented = pd.concat([df_real, df_synthetic], ignore_index=True)
    
    # Save combined output file
    df_augmented.to_csv(output_csv, index=False)

    print("==========================================")
    print(f"💾 Saved Output Dataset: '{output_csv}'")
    print(f"📊 Summary: {len(df_real)} Real + {num_synthetic_rows} Synthetic = {len(df_augmented)} Total Records")
    print("==========================================\n")

if __name__ == "__main__":
    generate_synthetic_data(
        input_csv="./Preprocessed/framingham_clean_dropping.csv",               # Path to your clean preprocessed CSV
        output_csv="./preprocessed_augmented.csv",  # Output CSV file
        num_synthetic_rows=1500                     # Number of synthetic rows to append
    )