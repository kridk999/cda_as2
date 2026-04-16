from pathlib import Path
import pandas as pd


# -----------------------------
# File-specific processors
# -----------------------------
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


# -----------------------------
# Process one phase folder
# -----------------------------
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


# -----------------------------
# Build dataframe for a phase
# -----------------------------
def build_dataframe(base_path, phase_name):
    rows = []

    for phase_path in base_path.rglob(phase_name):
        rows.append(process_phase_folder(phase_path))

    df = pd.DataFrame(rows)

    # Optional: sort rounds numerically
    df["round_num"] = df["round"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["D1", "ID", "round_num"])

    return df


# -----------------------------
# Compute stats from phase1
# -----------------------------
def compute_phase1_stats(df):
    feature_cols = ["BVP_mean", "EDA_mean", "HR_mean", "TEMP_mean"]

    stats = (
        df.groupby(["D1", "ID"])[feature_cols]
        .agg(["mean", "std"])
    )

    # Flatten column names
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]

    return stats.reset_index()


# -----------------------------
# Apply standardization
# -----------------------------
def apply_standardization(df, stats):
    feature_cols = ["BVP_mean", "EDA_mean", "HR_mean", "TEMP_mean"]

    # Merge stats into dataframe
    df = df.merge(stats, on=["D1", "ID"], how="left")

    for col in feature_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"

        df[col + "_z"] = (
            (df[col] - df[mean_col]) /
            df[std_col].replace(0, 1)
        )

    return df


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    base_path = Path("assets/data/dataset")

    # ---- Phase 1 ----
    df_phase1 = build_dataframe(base_path, "phase1")
    stats = compute_phase1_stats(df_phase1)

    df_phase1 = apply_standardization(df_phase1, stats)

    # ---- Phase 2 ----
    df_phase2 = build_dataframe(base_path, "phase2")
    df_phase2 = apply_standardization(df_phase2, stats)

    # ---- Save results ----
    df_phase1.to_csv("assets/data/phase1_processed.csv", index=False)
    df_phase2.to_csv("assets/data/phase2_processed.csv", index=False)


if __name__ == "__main__":
    main()