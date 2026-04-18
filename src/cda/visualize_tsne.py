import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

def run_tsne_visualization(phase1_path, phase2_path, nu, perplexity):
    df_phase1 = pd.read_csv(phase1_path)
    df_phase2 = pd.read_csv(phase2_path)


    feature_cols = ['HR_standardized', 'EDA_standardized', 'BVP_standardized', 'TEMP_standardized']
    

    X1 = df_phase1[feature_cols].values
    X2 = df_phase2[feature_cols].values
    
    X_combined = np.vstack((X1, X2))
    

    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    

    print("Running t-SNE... this might take a moment.")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_2d = tsne.fit_transform(X_combined_scaled)
    

    n_phase1 = len(X1)
    X1_2d = X_2d[:n_phase1]
    X2_2d = X_2d[n_phase1:]
    

    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    oc_svm.fit(X1_2d)


    plt.figure(figsize=(10, 7))
    

    xx, yy = np.meshgrid(np.linspace(X_2d[:, 0].min() - 5, X_2d[:, 0].max() + 5, 500),
                         np.linspace(X_2d[:, 1].min() - 5, X_2d[:, 1].max() + 5, 500))
    

    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    

    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r, alpha=0.8)
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='aliceblue', alpha=0.8)
    

    plt.contour(xx, yy, Z, levels=[0], linewidths=3, colors='red')


    p1 = plt.scatter(X1_2d[:, 0], X1_2d[:, 1], c='green', s=80, edgecolors='black', label="training samples (Phase 1)")
    p2 = plt.scatter(X2_2d[:, 0], X2_2d[:, 1], c='red', s=80, edgecolors='black', marker='>', label="test samples (Phase 2)")


    boundary_line = Line2D([0], [0], color='red', linewidth=3)


    plt.title("t-SNE Projection of Phase 1 vs Phase 2 with One-Class SVM Boundary")
    plt.legend([boundary_line, p1, p2],
               ["learned boundary", "training samples", "test samples"],
               loc="upper left",
               framealpha=1.0, 
               edgecolor="black",
               fontsize=12)
    
    plt.xlim((X_2d[:, 0].min() - 5, X_2d[:, 0].max() + 5))
    plt.ylim((X_2d[:, 1].min() - 5, X_2d[:, 1].max() + 5))
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create t-SNE 2D plot of Phase 1 & 2 mapped with an OC-SVM boundary.")
    parser.add_argument("--phase1", type=str, default="assets/data/phase1_processed.csv", 
                        help="Path to the Phase 1 processed CSV data.")
    parser.add_argument("--phase2", type=str, default="assets/data/phase2_processed.csv", 
                        help="Path to the Phase 2 processed CSV data.")
    parser.add_argument("--nu", type=float, default=0.05, 
                        help="An upper bound on the fraction of training errors (nu parameter for SVM).")
    parser.add_argument("--perplexity", type=int, default=30, 
                        help="Perplexity parameter for t-SNE.")
    
    args = parser.parse_args()
    
    run_tsne_visualization(
        phase1_path=args.phase1, 
        phase2_path=args.phase2, 
        nu=args.nu, 
        perplexity=args.perplexity
    )