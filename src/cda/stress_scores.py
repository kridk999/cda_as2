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


stress_diff = phase_2["stress_score"] - phase_1["stress_score"]/2


phase_2["stressed"] = stress_diff >= 3


num_stressed = phase_2["stressed"].sum()
total = num_stressed/len(phase_2)

print(f"Number of stressed individuals in phase 2: {total}")

phase_2[["Individual", "stressed"]].to_csv("assets/data/phase_2_stress_labels.csv", index=False)

# SVM labels
svm_labels = pd.read_csv("assets/data/SVM_labels.csv", header=0)

# Confusion matrix
from sklearn.metrics import confusion_matrix
y_true = phase_2["stressed"].astype(int)
y_pred = svm_labels["stressed"].astype(int)
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)