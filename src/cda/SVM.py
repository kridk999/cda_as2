import argparse
import pandas as pd
import numpy as np
import os
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def extract_time_series_features(df, window_size=10):
    """
    Extracts HR, EDA, and TEMP features while incorporating time-series 
    aspects via rolling means and rolling standard deviations.
    BVP is excluded due to noise.
    """
    df_feats = df.copy()
    base_cols = ['HR_standardized', 'EDA_standardized', 'TEMP_standardized']
    
    for col in base_cols:
        df_feats[f'{col}_raw'] = df_feats[col]
        
        df_feats[f'{col}_mean'] = df_feats.groupby(['D1', 'ID'])[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).mean())
        
        df_feats[f'{col}_std'] = df_feats.groupby(['D1', 'ID'])[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).std().fillna(0))
    
    feature_cols = [f'{col}_raw' for col in base_cols] + \
                   [f'{col}_mean' for col in base_cols] + \
                   [f'{col}_std' for col in base_cols]           
    return df_feats[feature_cols].values


def run_global_anomaly_detection(phase1_path, phase2_path, nu, window_size=10, plot=False):
    """
    Loads Phase 1 and Phase 2 data, trains a SINGLE One-Class SVM on all Phase 1 data, 
    and evaluates outlier ratios.
    """
    df_phase1 = pd.read_csv(phase1_path)
    df_phase2 = pd.read_csv(phase2_path)
    
    # Train 
    X_train_global = extract_time_series_features(df_phase1, window_size=window_size)
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    oc_svm.fit(X_train_global)
    
    # Predict on train ( for statistics only )
    preds_train_global = oc_svm.predict(X_train_global)
    train_total_samples = len(preds_train_global)
    train_outlier_count = np.sum(preds_train_global == -1)
    train_outlier_ratio = train_outlier_count / train_total_samples if train_total_samples > 0 else 0

    # Predict on test set
    X_test_global = extract_time_series_features(df_phase2, window_size=window_size)
    preds_global = oc_svm.predict(X_test_global)
    df_phase2['SVM_Prediction'] = preds_global

    test_total_samples = len(preds_global)
    test_outlier_count = np.sum(preds_global == -1)
    test_outlier_ratio = test_outlier_count / test_total_samples if test_total_samples > 0 else 0
    
    global_results = {
        'Train_Total_Samples': train_total_samples,
        'Train_Outlier_Count': train_outlier_count,
        'Train_Outlier_Ratio': train_outlier_ratio,
        'Test_Total_Samples': test_total_samples,
        'Test_Outlier_Count': test_outlier_count,
        'Test_Outlier_Ratio': test_outlier_ratio
    }


    if plot:
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train_global)
        X_test_2d = pca.transform(X_test_global) if len(X_test_global) > 0 else np.empty((0,2))
        
        plt.figure(figsize=(10, 6), dpi=150)
        
        color_normal = '#4C72B0'
        color_outlier = '#DD8452'
        
        plt.scatter(X_train_2d[preds_train_global == 1, 0], X_train_2d[preds_train_global == 1, 1], 
                    c=color_normal, marker='o', s=15, edgecolors='none', label='Phase 1 - Normal', alpha=0.5)
        plt.scatter(X_train_2d[preds_train_global == -1, 0], X_train_2d[preds_train_global == -1, 1], 
                    c=color_outlier, marker='o', s=15, edgecolors='none', label='Phase 1 - Outlier', alpha=0.5)
        
        if len(X_test_global) > 0:
            plt.scatter(X_test_2d[preds_global == 1, 0], X_test_2d[preds_global == 1, 1], 
                        c=color_normal, marker='D', s=45, edgecolors='white', linewidth=1, label='Phase 2 - Normal', alpha=0.9)
            plt.scatter(X_test_2d[preds_global == -1, 0], X_test_2d[preds_global == -1, 1], 
                        c=color_outlier, marker='D', s=45, edgecolors='white', linewidth=1, label='Phase 2 - Outlier', alpha=0.9)

        plt.title('Global OC-SVM Anomalies (PCA Projection)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)
        
        plt.legend(loc='best', markerscale=1.5, frameon=True, edgecolor='lightgray', borderpad=1)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    # --------------------------------------

    return global_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate One-Class SVM for anomaly detection.")
    parser.add_argument("--phase1", type=str, default="assets/data/phase1_processed.csv", 
                        help="Path to the Phase 1 processed CSV data.")
    parser.add_argument("--phase2", type=str, default="assets/data/phase2_processed.csv", 
                        help="Path to the Phase 2 processed CSV data.")
    parser.add_argument("--nu", type=float, default=0.10, 
                        help="An upper bound on the fraction of training errors (nu parameter for SVM).")    
    parser.add_argument("--plot", action="store_true", 
                        help="Add this flag to display the time-series visualization for the first individual.") 
    parser.add_argument("--window_size", type=int, default=3, 
                        help="Number of timesteps for the rolling windows to capture time-series trends.")
    
    args = parser.parse_args()
    
    global_results = run_global_anomaly_detection(
        phase1_path=args.phase1, 
        phase2_path=args.phase2, 
        nu=args.nu, 
        window_size=args.window_size,
        plot=args.plot  
    )
    
    # --- INTERPRETATION STATISTICS ---
    print("\n" + "="*50)
    print(" GLOBAL ANOMALY DETECTION STATISTICS ".center(50, "="))
    print("="*50)
    
    train_samples = global_results['Train_Total_Samples']
    test_samples = global_results['Test_Total_Samples']
    
    if test_samples > 0 and train_samples > 0:
        train_outliers = global_results['Train_Outlier_Count']
        train_ratio = global_results['Train_Outlier_Ratio']
        
        test_outliers = global_results['Test_Outlier_Count']
        test_ratio = global_results['Test_Outlier_Ratio']
        
        print(f"--- TRAINING DATA (Phase 1) ---")
        print(f"Total Samples Evaluated:    {train_samples}")
        print(f"Anomalous Samples Flagged:  {train_outliers} ({(train_ratio * 100):.2f}%)")
        print(f"(Note: Expected ~ {args.nu * 100:.1f} % based on nu parameter)")
        
        print(f"\n--- TESTING DATA (Phase 2) ---")
        print(f"Total Samples Evaluated:    {test_samples}")
        print(f"Anomalous Samples Flagged:  {test_outliers} ({(test_ratio * 100):.2f}%)")
    else:
        print("Data missing to calculate complete statistics.")
    print("=" * 50 + "\n")

