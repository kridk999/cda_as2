import pandas as pd 

df = pd.read_csv("assets/data/HR_data.csv", header=0)

cols = [
    "Round","Phase","Individual","Puzzler","Frustrated","Cohort",
    "upset","hostile","alert","ashamed","inspired","nervous",
    "attentive","afraid","active","determined"
]


df_clean = df[cols].copy()

# Split by phase
phase_1 = df_clean[df_clean["Phase"] == "phase1"].reset_index(drop=True)
phase_2 = df_clean[df_clean["Phase"] == "phase2"].reset_index(drop=True)

phase_1["stress_score"] = (
    phase_1["Frustrated"]/2 +
    phase_1["upset"]
)

phase_2["stress_score"] = (
    phase_2["Frustrated"]/2 +
    phase_2["upset"]
)

# see if phase_2 stress score is >3 hghter than phase_1 stress score
stress_diff = phase_2["stress_score"] - phase_1["stress_score"]


phase_2["stressed"] = stress_diff >= 3

phase_2[["Individual", "stressed"]].to_csv("assets/data/phase_2_stress_labels.csv", index=False)