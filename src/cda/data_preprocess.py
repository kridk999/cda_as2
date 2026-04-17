from pathlib import Path
import pandas as pd

# Processors for each signal type 
def process_bvp(path):
    df = pd.read_csv(path)
    return {"BVP_mean": df.iloc[:, 1].mean()}

def process_eda(path):
    df = pd.read_csv(path)
    return {"EDA_mean": df.iloc[:, 1].mean()}

def process_hr(path):
    df = pd.read_csv(path)
    return {"HR_mean": df.iloc[:, 1].mean()}

def process_temp(path):
    df = pd.read_csv(path)
    return {"TEMP_mean": df.iloc[:, 1].mean()}

PROCESSORS = {
    "bvp": process_bvp,
    "eda": process_eda,
    "hr": process_hr,
    "temp": process_temp
}


# Collecting data from each D1 for each person for the 4 rounds for phase 1 and 2
def process_phase_folder(phase_path):
    parts = phase_path.parts

    d1 = next(p for p in parts if p.startswith("D1_"))
    id_ = next(p for p in parts if p.startswith("ID_"))
    round_ = next(p for p in parts if p.startswith("round_"))

    row = {
        "D1": d1,
        "ID": id_,
        "round": round_
    }

    for name, func in PROCESSORS.items():
        file = phase_path / f"{name.upper()}.csv"
        if file.exists():
            row.update(func(file))

    return row



# Build dataframe for the phase 1 and phase 2 datasets
def build_dataframe(base_path, phase_name):
    rows = []

    for phase_path in base_path.rglob(phase_name):
        rows.append(process_phase_folder(phase_path))

    df = pd.DataFrame(rows)

    # Sort rounds numerically
    df["round"] = df["round"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["D1", "ID", "round"])

    return df


# Compute mean and standard deviation from phase 1 dataset for each D1 and person across the 4 rounds
def compute_phase1_stats(df):
    feature_cols = ["BVP_mean", "EDA_mean", "HR_mean", "TEMP_mean"]

    stats = (
        df.groupby(["D1", "ID"])[feature_cols]
        .agg(["mean", lambda x: x.std(ddof=0)])
    )

    stats.columns = [f"{col}_{stat}" for col in feature_cols for stat in ["mean", "std"]]

    return stats.reset_index()


# Apply standardization for each person and D1 using the computed mean and std from phase 1
def apply_standardization(df, stats):
    feature_cols = ["BVP", "EDA", "HR", "TEMP"]

    df = df.merge(stats, on=["D1", "ID"], how="left")

    for col in feature_cols:
        df[f"{col}_standardized"] = (
            (df[f"{col}_mean"] - df[f"{col}_mean_mean"]) /
            df[f"{col}_mean_std"].replace(0, 1)
        )

    return df[[
        "D1",
        "ID",
        "round",
        "BVP_standardized",
        "EDA_standardized",
        "HR_standardized",
        "TEMP_standardized"
    ]]

# Main pipeline
def main():
    base_path = Path("assets/data/dataset") # dataset folder path

    # Phase 1
    df_phase1 = build_dataframe(base_path, "phase1")
    stats = compute_phase1_stats(df_phase1)

    df_phase1 = apply_standardization(df_phase1, stats)

    # Phase 2 
    df_phase2 = build_dataframe(base_path, "phase2")
    df_phase2 = apply_standardization(df_phase2, stats)

    # Save results
    df_phase1.to_csv("assets/data/phase1_processed.csv", index=False)
    df_phase2.to_csv("assets/data/phase2_processed.csv", index=False)


if __name__ == "__main__":
    main()