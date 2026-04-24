import argparse
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_feature_matrix(df_subset):
    """
    Takes a DataFrame subset and extracts the standardized sensor columns
    into a 2D numpy array of shape (N_samples, 4_features).
    """
    # Simply select the existing standardized columns
    feature_cols = [
        'HR_standardized', 
        'EDA_standardized', 
        'BVP_standardized', 
        #'TEMP_standardized'
    ]
    
    return df_subset[feature_cols].values


def run_global_anomaly_detection(phase1_path, phase2_path, nu, plot=False, plot_tsne=False):
    """
    Loads Phase 1 and Phase 2 data, trains a SINGLE One-Class SVM on all Phase 1 data, 
    and evaluates outlier ratios.
    """
    df_phase1 = pd.read_csv(phase1_path)
    df_phase2 = pd.read_csv(phase2_path)
    
    # Train 
    X_train_global = extract_feature_matrix(df_phase1)
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    oc_svm.fit(X_train_global)
    
    # Predict on train ( for statistics only )
    preds_train_global = oc_svm.predict(X_train_global)
    df_phase1['SVM_Prediction'] = preds_train_global    # Add predictions to Phase 1 DataFrame
    
    train_total_samples = len(preds_train_global)
    train_outlier_count = np.sum(preds_train_global == -1)
    train_outlier_ratio = train_outlier_count / train_total_samples if train_total_samples > 0 else 0

    # Predict on test set
    X_test_global = extract_feature_matrix(df_phase2)
    preds_global = oc_svm.predict(X_test_global)
    df_phase2['SVM_Prediction'] = preds_global

    test_total_samples = len(preds_global)
    test_outlier_count = np.sum(preds_global == -1)
    test_outlier_ratio = test_outlier_count / test_total_samples if test_total_samples > 0 else 0
    
    train_individual_stats = {}
    if 'ID' in df_phase1.columns:
        for person_id, group in df_phase1.groupby('ID'):
            total = len(group)
            outliers = np.sum(group['SVM_Prediction'] == -1)
            train_individual_stats[person_id] = {
                'Total_Samples': total,
                'Outlier_Count': outliers,
                'Outlier_Ratio': outliers / total if total > 0 else 0
            }

    test_individual_stats = {}
    if 'ID' in df_phase2.columns:
        for person_id, group in df_phase2.groupby('ID'):
            total = len(group)
            outliers = np.sum(group['SVM_Prediction'] == -1)
            test_individual_stats[person_id] = {
                'Total_Samples': total,
                'Outlier_Count': outliers,
                'Outlier_Ratio': outliers / total if total > 0 else 0
            }
            
    global_results = {
        'Train_Total_Samples': train_total_samples,
        'Train_Outlier_Count': train_outlier_count,
        'Train_Outlier_Ratio': train_outlier_ratio,
        'Test_Total_Samples': test_total_samples,
        'Test_Outlier_Count': test_outlier_count,
        'Test_Outlier_Ratio': test_outlier_ratio,
        'Train_Individual_Stats': train_individual_stats,
        'Test_Individual_Stats': test_individual_stats
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

    return global_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate One-Class SVM for anomaly detection.")
    parser.add_argument("--phase1", type=str, default="assets/data/phase1_and_3_processed.csv", 
                        help="Path to the Phase 1 processed CSV data.")
    parser.add_argument("--phase2", type=str, default="assets/data/phase2_processed.csv", 
                        help="Path to the Phase 2 processed CSV data.")
    parser.add_argument("--nu", type=float, default=0.10, 
                        help="An upper bound on the fraction of training errors (nu parameter for SVM).")    
    parser.add_argument("--plot", action="store_true", 
                        help="Add this flag to display the time-series visualization for the first individual.") 
    parser.add_argument("--plot_tsne", action="store_true", default=False,
                        help="Add this flag to display t-SNE visualization of the SVM results.")
    parser.add_argument("--save_as_csv", action="store_true", default=False,
                        help="Add this flag to save the global anomaly detection results as a CSV file.")
    args = parser.parse_args()
    
    global_results = run_global_anomaly_detection(
        phase1_path=args.phase1, 
        phase2_path=args.phase2, 
        nu=args.nu, 
        plot=args.plot,
        plot_tsne=args.plot_tsne
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
        
        print("\n--- PHASE 2 INDIVIDUAL STATISTICS ---")
        test_ind_stats = global_results.get('Test_Individual_Stats', {})
        for person_id, stats in test_ind_stats.items():
            print(f"{person_id: <8} | Samples: {stats['Total_Samples']: <4} | Outliers: {stats['Outlier_Count']: <4} | Ratio: {stats['Outlier_Ratio']*100:.2f}%")
    else:
        print("Data missing to calculate complete statistics.")
    print("=" * 50 + "\n")

    if args.save_as_csv:
        # Build a list of rows to save to CSV including both global and individual stats
        csv_rows = []
        
        # Add Global rows
        csv_rows.append({
            'Phase': 'Train', 'ID': 'Global', 
            'Total_Samples': train_samples, 'Outlier_Count': train_outliers, 'Outlier_Ratio': train_ratio
        })
        csv_rows.append({
            'Phase': 'Test', 'ID': 'Global', 
            'Total_Samples': test_samples, 'Outlier_Count': test_outliers, 'Outlier_Ratio': test_ratio
        })
        
        # Add Individual Train rows
        train_ind_stats = global_results.get('Train_Individual_Stats', {})
        for pid, stats in train_ind_stats.items():
            csv_rows.append({
                'Phase': 'Train', 'ID': pid, 
                'Total_Samples': stats['Total_Samples'], 'Outlier_Count': stats['Outlier_Count'], 'Outlier_Ratio': stats['Outlier_Ratio']
            })
            
        # Add Individual Test rows
        test_ind_stats = global_results.get('Test_Individual_Stats', {})
        for pid, stats in test_ind_stats.items():
            csv_rows.append({
                'Phase': 'Test', 'ID': pid, 
                'Total_Samples': stats['Total_Samples'], 'Outlier_Count': stats['Outlier_Count'], 'Outlier_Ratio': stats['Outlier_Ratio']
            })
            
        # Save the results to a CSV file
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv('svm_global_anomaly_detection_results.csv', index=False)
        print("Results saved to svm_global_anomaly_detection_results.csv")