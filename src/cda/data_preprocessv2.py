import pandas as pd

df = pd.read_csv("assets/data/HR_data.csv", header=0)

cols = [
    'HR_TD_Mean', 'HR_TD_Median', 'HR_TD_std', 'HR_TD_Min',
    'HR_TD_Max', 'HR_TD_AUC', 'HR_TD_Kurtosis', 'HR_TD_Skew',
    'HR_TD_Slope_min', 'HR_TD_Slope_max', 'HR_TD_Slope_mean', 'HR_TD_Slope',
    'TEMP_TD_Mean', 'TEMP_TD_Median', 'TEMP_TD_std', 'TEMP_TD_Min',
    'TEMP_TD_Max', 'TEMP_TD_AUC', 'TEMP_TD_Kurtosis', 'TEMP_TD_Skew',
    'TEMP_TD_Slope_min', 'TEMP_TD_Slope_max', 'TEMP_TD_Slope_mean',
    'TEMP_TD_Slope', 'EDA_TD_P_Mean', 'EDA_TD_P_Median', 'EDA_TD_P_std',
    'EDA_TD_P_Min', 'EDA_TD_P_Max', 'EDA_TD_P_AUC', 'EDA_TD_P_Kurtosis',
    'EDA_TD_P_Skew', 'EDA_TD_P_Slope_min', 'EDA_TD_P_Slope_max',
    'EDA_TD_P_Slope_mean', 'EDA_TD_P_Slope', 'EDA_TD_T_Mean',
    'EDA_TD_T_Median', 'EDA_TD_T_std', 'EDA_TD_T_Min', 'EDA_TD_T_Max',
    'EDA_TD_T_AUC', 'EDA_TD_T_Kurtosis', 'EDA_TD_T_Skew',
    'EDA_TD_T_Slope_min', 'EDA_TD_T_Slope_max', 'EDA_TD_T_Slope_mean',
    'EDA_TD_T_Slope', 'EDA_TD_P_Peaks', 'EDA_TD_P_RT', 'EDA_TD_P_ReT',
    'Round', 'Phase', 'Individual'
]

feature_cols = [
    'HR_TD_Mean', 'HR_TD_Median', 'HR_TD_std', 'HR_TD_Min',
    'HR_TD_Max', 'HR_TD_AUC', 'HR_TD_Kurtosis', 'HR_TD_Skew',
    'HR_TD_Slope_min', 'HR_TD_Slope_max', 'HR_TD_Slope_mean', 'HR_TD_Slope',
    'TEMP_TD_Mean', 'TEMP_TD_Median', 'TEMP_TD_std', 'TEMP_TD_Min',
    'TEMP_TD_Max', 'TEMP_TD_AUC', 'TEMP_TD_Kurtosis', 'TEMP_TD_Skew',
    'TEMP_TD_Slope_min', 'TEMP_TD_Slope_max', 'TEMP_TD_Slope_mean',
    'TEMP_TD_Slope', 'EDA_TD_P_Mean', 'EDA_TD_P_Median', 'EDA_TD_P_std',
    'EDA_TD_P_Min', 'EDA_TD_P_Max', 'EDA_TD_P_AUC', 'EDA_TD_P_Kurtosis',
    'EDA_TD_P_Skew', 'EDA_TD_P_Slope_min', 'EDA_TD_P_Slope_max',
    'EDA_TD_P_Slope_mean', 'EDA_TD_P_Slope', 'EDA_TD_T_Mean',
    'EDA_TD_T_Median', 'EDA_TD_T_std', 'EDA_TD_T_Min', 'EDA_TD_T_Max',
    'EDA_TD_T_AUC', 'EDA_TD_T_Kurtosis', 'EDA_TD_T_Skew',
    'EDA_TD_T_Slope_min', 'EDA_TD_T_Slope_max', 'EDA_TD_T_Slope_mean',
    'EDA_TD_T_Slope', 'EDA_TD_P_Peaks', 'EDA_TD_P_RT', 'EDA_TD_P_ReT'
]

df_clean = df[cols].copy()

df_clean = df_clean.dropna(subset=feature_cols)

df_phase1 = df_clean[df_clean["Phase"] == "phase1"].reset_index(drop=True)
df_phase2 = df_clean[df_clean["Phase"] == "phase2"].reset_index(drop=True)

def compute_phase1_stats(df_phase1):
    means = df_phase1.groupby("Individual")[feature_cols].mean()
    stds = df_phase1.groupby("Individual")[feature_cols].agg(lambda x: x.std(ddof=0))
    stds = stds.replace(0, 1).fillna(1)
    return means, stds

def apply_standardization(df, means, stds):
    df = df.copy()

    for col in feature_cols:
        df[col] = (
            df[col] - df["Individual"].map(means[col])
        ) / df["Individual"].map(stds[col])

    return df

phase1_means, phase1_stds = compute_phase1_stats(df_phase1)

df_phase1_std = apply_standardization(df_phase1, phase1_means, phase1_stds)
df_phase2_std = apply_standardization(df_phase2, phase1_means, phase1_stds)

df_phase1_std = df_phase1_std.set_index(["Individual", "Round"]).sort_index()
df_phase2_std = df_phase2_std.set_index(["Individual", "Round"]).sort_index()

df_phase1_std.to_csv("assets/data/phase1_extra_features_processed.csv")
df_phase2_std.to_csv("assets/data/phase2_extra_features_processed.csv")