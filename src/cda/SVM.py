import pandas as pd
import numpy as np
import os
from dummy_dataset_generator import generate_dummy_dataframes
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.join("assets", "dummy_dataset")
NUM_COHORTS = 6       
NUM_IDS = 3           
NUM_ROUNDS = 4        
NUM_PHASES = 3        
VECTOR_LENGTH = 10    

dfs = generate_dummy_dataframes()
    
df_phase1 = dfs["Phase_1"]
df_phase2 = dfs["Phase_2"]
df_phase3 = dfs["Phase_3"]

# 1. Combine Phase 1 and Phase 3 to form the baseline/training set
df_baseline = pd.concat([df_phase1, df_phase3], ignore_index=True)

results = []

def extract_feature_matrix(df_subset):
    """
    Takes a DataFrame subset (for a specific person) and extracts the 
    sensor vectors into a 2D numpy array of shape (N_samples, 4_features)
    """
    # np.concatenate joins the lists across all rounds into a single long array
    hr = np.concatenate(df_subset['HR_vector'].values)
    eda = np.concatenate(df_subset['EDA_vector'].values)
    bvp = np.concatenate(df_subset['BVP_vector'].values)
    temp = np.concatenate(df_subset['TEMP_vector'].values)
    
    # Stack columns side-by-side: shape becomes (N_samples, 4)
    return np.column_stack((hr, eda, bvp, temp))

# 2. Iterate on an individual basis (Grouping by Cohort and Person)
for (cohort, person), baseline_group in df_baseline.groupby(['Cohort', 'Person']):
    
    # Extract matching Phase 2 data for testing
    test_group = df_phase2[(df_phase2['Cohort'] == cohort) & (df_phase2['Person'] == person)]
    
    # Extract feature matrices
    X_train = extract_feature_matrix(baseline_group)
    X_test = extract_feature_matrix(test_group)
    
    # 3. Normalize the data
    scaler = StandardScaler()
    # Fit the scaler ONLY on the baseline data, then transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train the One-Class SVM detector
    # 'nu' is an upper bound on the fraction of training errors (expected outliers in training set)
    # We set it slightly above 0 to prevent harsh overfitting.
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    oc_svm.fit(X_train_scaled)
    
    # 5. Predict on Phase 2 data
    # Output is +1 for inliers (normal) and -1 for outliers (anomalous)
    preds = oc_svm.predict(X_test_scaled)
    
    # Calculate the percentage of samples in Phase 2 flagged as outliers
    total_samples = len(preds)
    outlier_count = np.sum(preds == -1)
    outlier_ratio = outlier_count / total_samples
    
    # We can define a threshold (e.g., > 30% anomalous samples means Phase 2 is an outlier phase)
    is_outlier_phase = outlier_ratio > 0.30
    
    results.append({
        'Cohort': cohort,
        'Person': person,
        'Phase2_Outlier_Ratio': outlier_ratio,
        'Phase2_Is_Outlier': is_outlier_phase
    })

# Convert results to a DataFrame for easy viewing
results_df = pd.DataFrame(results)

print("--- Outlier Detection Results per Individual ---")
print(results_df.head(10))
