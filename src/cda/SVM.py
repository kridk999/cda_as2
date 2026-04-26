import argparse
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_feature_matrix(df_subset):
    """
    Takes a DataFrame subset and extracts the feature columns
    by ignoring metadata columns like Individual, Round, and Phase.
    """
    # Columns that shouldn't be fed to the SVM
    exclude_cols = ['Individual', 'Round', 'Phase', 'ID'] 
    
    # Keep only the feature columns
    feature_cols = [col for col in df_subset.columns if col not in exclude_cols]
    
    return df_subset[feature_cols].values


def run_global_anomaly_detection(phase1_path, phase2_path, nu, plot=False):
    """
    Loads Phase 1 and Phase 2 data, trains a One-Class SVM on all Phase 1 data, 
    and evaluates outliers.
    """
    df_phase1 = pd.read_csv(phase1_path)
    df_phase2 = pd.read_csv(phase2_path)
    
    
    X_train_global = extract_feature_matrix(df_phase1)
    
    pca = PCA(n_components=0.9, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_global)
    feature_names = [col for col in df_phase1.columns if col not in ['Individual', 'Round', 'Phase', 'ID']]
  
    print(len(feature_names), "features reduced to", X_train_pca.shape[1], "principal components.")
    
    oc_svm = OneClassSVM(kernel='rbf',gamma=0.01, nu=nu)
    oc_svm.fit(X_train_pca)
  
    preds_train_global = oc_svm.predict(X_train_pca)
    df_phase1['SVM_Prediction'] = preds_train_global    
    train_total_samples = len(preds_train_global)
    train_outlier_count = np.sum(preds_train_global == -1)
    train_outlier_ratio = train_outlier_count / train_total_samples if train_total_samples > 0 else 0

    X_test_global = extract_feature_matrix(df_phase2)
    X_test_pca = pca.transform(X_test_global)
    preds_global = oc_svm.predict(X_test_pca)
    df_phase2['SVM_Prediction'] = preds_global

    test_total_samples = len(preds_global)
    test_outlier_count = np.sum(preds_global == -1)
    test_outlier_ratio = test_outlier_count / test_total_samples if test_total_samples > 0 else 0
    
    # --- REWRITTEN global_results BLOCK ---
    id_col1 = 'Individual' if 'Individual' in df_phase1.columns else 'ID'
    id_col2 = 'Individual' if 'Individual' in df_phase2.columns else 'ID'
    
    # Extract the required columns for each phase
    res1 = df_phase1[[id_col1, 'Phase', 'SVM_Prediction']].copy()
    res1.rename(columns={id_col1: 'Individual'}, inplace=True)
    
    res2 = df_phase2[[id_col2, 'Phase', 'SVM_Prediction']].copy()
    res2.rename(columns={id_col2: 'Individual'}, inplace=True)
    
    # Combine them into a single structure (DataFrame)
    global_results = pd.concat([res1, res2], ignore_index=True)
    # ----------------------------------------

    if plot:
        pca_plot = PCA(n_components=2)
        X_train_2d = pca_plot.fit_transform(X_train_pca)
        X_test_2d = pca_plot.transform(X_test_pca) if len(X_test_pca) > 0 else np.empty((0,2))
        
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

    return global_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate One-Class SVM for anomaly detection.")
    parser.add_argument("--phase1", type=str, default="assets\\data\\phase1_extra_features_processed.csv", 
                        help="Path to the Phase 1 processed CSV data.")
    parser.add_argument("--phase2", type=str, default="assets\\data\\phase2_extra_features_processed.csv", 
                        help="Path to the Phase 2 processed CSV data.")
    parser.add_argument("--nu", type=float, default=0.10, 
                        help="An upper bound on the fraction of training errors (nu parameter for SVM).")    
    parser.add_argument("--plot", action="store_true", 
                        help="Add this flag to display the time-series visualization for the first individual.") 
    parser.add_argument("--save_as_csv", action="store_true", default=False,
                        help="Add this flag to save the global anomaly detection results as a CSV file.")
    args = parser.parse_args()
    
    global_results = run_global_anomaly_detection(
        phase1_path=args.phase1, 
        phase2_path=args.phase2, 
        nu=args.nu, 
        plot=args.plot
    )
    
    print("\n" + "="*50)
    print(" GLOBAL ANOMALY DETECTION STATISTICS ".center(50, "="))
    print("="*50)
    
    print(global_results.value_counts(subset=['Phase', 'SVM_Prediction']).to_frame(name='Count').reset_index())

    if args.save_as_csv:
        # Save the DataFrame directly to a CSV file
        global_results.to_csv('svm_results.csv', index=False)
        print("Results saved to svm_results.csv")