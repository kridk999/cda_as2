import argparse
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def extract_feature_matrix(df_subset):
    """
    Takes a DataFrame subset (for a specific person) and extracts the 
    sensor values into a 2D numpy array of shape (N_samples, 4_features)
    """
    hr = df_subset['HR_standardized'].values
    eda = df_subset['EDA_standardized'].values
    bvp = df_subset['BVP_standardized'].values
    temp = df_subset['TEMP_standardized'].values
    
    return np.column_stack((hr, eda, bvp, temp))

def run_anomaly_detection(phase1_path, phase2_path, nu, threshold):
    """
    Loads Phase 1 and Phase 2 data, trains a One-Class SVM per individual 
    on Phase 1, and evaluates outlier ratios on Phase 2.
    """
    df_phase1 = pd.read_csv(phase1_path)
    df_phase2 = pd.read_csv(phase2_path)

    results = []

    # iterate on an individual basis on phase 1 data
    for (d1, id_), baseline_group in df_phase1.groupby(['D1', 'ID']):
        
        # test group for the same individual in phase 2
        test_group = df_phase2[(df_phase2['D1'] == d1) & (df_phase2['ID'] == id_)]
        
            
        # training and test data for the current individual
        X_train = extract_feature_matrix(baseline_group)
        X_test = extract_feature_matrix(test_group)
        
        # normalize data 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # fit one-class SVM
        oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
        oc_svm.fit(X_train_scaled)
        

        preds = oc_svm.predict(X_test_scaled)
        

        total_samples = len(preds)
        outlier_count = np.sum(preds == -1)
        outlier_ratio = outlier_count / total_samples
        
        is_outlier_phase = outlier_ratio > threshold
        
        results.append({
            'D1': d1,
            'ID': id_,
            'Phase2_Outlier_Ratio': outlier_ratio,
            'Phase2_Is_Outlier': is_outlier_phase
        })

    return pd.DataFrame(results)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate One-Class SVM for anomaly detection.")
    parser.add_argument("--phase1", type=str, default="assets/data/phase1_processed.csv", 
                        help="Path to the Phase 1 processed CSV data.")
    parser.add_argument("--phase2", type=str, default="assets/data/phase2_processed.csv", 
                        help="Path to the Phase 2 processed CSV data.")
    parser.add_argument("--nu", type=float, default=0.05, 
                        help="An upper bound on the fraction of training errors (nu parameter for SVM).")
    parser.add_argument("--threshold", type=float, default=0.30, 
                        help="Outlier ratio threshold above which a phase is considered anomalous.")
    
    args = parser.parse_args()
    
    results_df = run_anomaly_detection(
        phase1_path=args.phase1, 
        phase2_path=args.phase2, 
        nu=args.nu, 
        threshold=args.threshold
    )
    
    print("--- Outlier Detection Results per Individual ---")
    print(results_df.head(10))