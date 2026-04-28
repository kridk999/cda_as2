"""
SVM Diagnostic Script
=====================
Runs four checks to assess whether the One-Class SVM results are genuine
or driven by normalization artefact / hyperparameter sensitivity:

    1. PCA component analysis
    2. Nu parameter sweep
    3. Gamma parameter sweep
    4. Shuffle test (most important)

Usage:
    python svm_diagnostics.py \
        --phase1 assets/data/phase1_extra_features_processed.csv \
        --phase2 assets/data/phase2_extra_features_processed.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

DTU_RED    = "#990000"
DTU_ORANGE = "#FC7634"
DTU_GREEN  = "#008835"
DTU_BLUE   = "#2F3EEA"
DTU_GRAY   = "#999999"


def load_features(path):
    """Load CSV and drop metadata columns, return numpy array."""
    df = pd.read_csv(path)
    exclude = {"Individual", "Round", "Phase", "ID", "Cohort", "Puzzler"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols].values, feature_cols

def fit_pca(X_train, variance_threshold=0.90):
    """Fit PCA on Phase 1, return fitted PCA and transformed data."""
    pca = PCA(n_components=variance_threshold, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    return pca, X_train_pca


def pca_variance(X_train, feature_cols, ax_cum, ax_bar):
    """Plot cumulative and individual explained variance."""
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = np.searchsorted(cumvar, 0.90) + 1
    n_components_80 = np.searchsorted(cumvar, 0.80) + 1

    ax_cum.plot(range(1, len(cumvar) + 1), cumvar,
                color=DTU_ORANGE, marker='o', markersize=4, linewidth=1.5)
    ax_cum.axhline(0.90, color=DTU_RED,   linestyle='--', linewidth=1,
                   label=f'90% : {n_components_90} components')
    ax_cum.axhline(0.80, color=DTU_GREEN, linestyle='--', linewidth=1,
                   label=f'80% : {n_components_80} components')
    ax_cum.set_xlabel('Number of components')
    ax_cum.set_ylabel('Cumulative explained variance')
    ax_cum.set_title('PCA Cumulative Variance')
    ax_cum.legend(fontsize=9)
    ax_cum.set_xlim(1, min(30, len(cumvar)))
    ax_cum.grid(True, alpha=0.3)
    n_show = min(20, len(pca_full.explained_variance_ratio_))
    ax_bar.bar(range(1, n_show + 1),
               pca_full.explained_variance_ratio_[:n_show],
               color=DTU_BLUE, alpha=0.7)
    ax_bar.set_xlabel('Component index')
    ax_bar.set_ylabel('Individual explained variance')
    ax_bar.set_title('Individual Component Variance')
    ax_bar.grid(True, alpha=0.3, axis='y')

    print("Check 1: PCA Variance")
    print(f"Total features: {X_train.shape[1]}")
    print(f"Training observations: {X_train.shape[0]}")
    print(f"Components for 80% var: {n_components_80}")
    print(f"Components for 90% var: {n_components_90}")
    print(f"Obs / component (90%): {X_train.shape[0] / n_components_90:.1f} "
          f"(want > 5, ideally > 10)")
    if X_train.shape[0] / n_components_90 < 5:
        print("Warning: Very sparse, boundary will be unreliable")
    print()

    return n_components_90

def nu_sweep(X_train_pca, X_test_pca, self_report_rate, ax):
    """Sweep nu and plot Phase 1 vs Phase 2 outlier rates."""
    nu_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    p1_rates, p2_rates = [], []

    for nu in nu_values:
        svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
        svm.fit(X_train_pca)
        p1_rates.append(np.mean(svm.predict(X_train_pca) == -1))
        p2_rates.append(np.mean(svm.predict(X_test_pca)  == -1))

    ax.plot(nu_values, p1_rates, 'o-', color=DTU_GREEN,  label='Phase 1 outlier rate')
    ax.plot(nu_values, p2_rates, 'o-', color=DTU_RED,    label='Phase 2 outlier rate')
    ax.axhline(self_report_rate, color=DTU_GRAY, linestyle='--', linewidth=1,
               label=f'Self-report stress rate ({self_report_rate:.0%})')
    ax.set_xlabel('nu')
    ax.set_ylabel('Outlier rate')
    ax.set_title('Nu Parameter Sweep')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    print("Check 2: Nu Sweep")
    print(f"{'nu':>6}  {'Phase1':>8}  {'Phase2':>8}  {'Gap':>8}")
    for nu, r1, r2 in zip(nu_values, p1_rates, p2_rates):
        print(f"{nu:>6.2f}  {r1:>8.2%}  {r2:>8.2%}  {r2-r1:>+8.2%}")
    print()

def gamma_sweep(X_train_pca, X_test_pca, nu, ax):
    """Sweep gamma values and plot Phase 2 outlier rate."""
    gamma_labels = ['0.001', '0.01', '0.05', '0.1', 'scale', 'auto']
    gamma_values = [0.001,   0.01,   0.05,   0.1,   'scale', 'auto']
    p2_rates = []

    for gamma in gamma_values:
        svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        svm.fit(X_train_pca)
        p2_rates.append(np.mean(svm.predict(X_test_pca) == -1))

    bars = ax.bar(gamma_labels, p2_rates, color=DTU_ORANGE, alpha=0.8)
    ax.set_xlabel('gamma')
    ax.set_ylabel('Phase 2 outlier rate')
    ax.set_title(f'Gamma Sweep (nu={nu})')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, p2_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                f'{rate:.0%}', ha='center', va='bottom', fontsize=9)

    print("Check 3: Gamma Sweep")
    for label, rate in zip(gamma_labels, p2_rates):
        print(f"gamma={label:<8} Phase 2 outlier rate: {rate:.2%}")
    if max(p2_rates) - min(p2_rates) < 0.10:
        print("Warning: Outlier rate barely changes with gamma, problem is not kernel width")
    print()

def shuffle_test(X_train_pca, X_test_pca, nu, n_repeats, ax):
    """
    Randomly shuffle phase labels n_repeats times and record outlier rates.
    If shuffled rate is roughly equal to real rate, result is artefact not signal.
    """
    all_data = np.vstack([X_train_pca, X_test_pca])
    n_train  = len(X_train_pca)

    svm_real = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    svm_real.fit(X_train_pca)
    real_rate = np.mean(svm_real.predict(X_test_pca) == -1)

    shuffled_rates = []
    for _ in range(n_repeats):
        idx = np.random.permutation(len(all_data))
        X_tr = all_data[idx[:n_train]]
        X_te = all_data[idx[n_train:]]
        svm  = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
        svm.fit(X_tr)
        shuffled_rates.append(np.mean(svm.predict(X_te) == -1))

    shuffled_rates = np.array(shuffled_rates)
    mean_shuffled  = shuffled_rates.mean()
    std_shuffled   = shuffled_rates.std()

    ax.hist(shuffled_rates, bins=20, color=DTU_BLUE, alpha=0.7,
            label=f'Shuffled ({n_repeats} runs)\nmean={mean_shuffled:.2%}')
    ax.axvline(real_rate, color=DTU_RED, linewidth=2,
               label=f'Real Phase 2 rate={real_rate:.2%}')
    ax.axvline(mean_shuffled, color=DTU_BLUE, linewidth=1.5,
               linestyle='--', label=f'Shuffle mean={mean_shuffled:.2%}')
    ax.set_xlabel('Phase 2 outlier rate')
    ax.set_ylabel('Frequency')
    ax.set_title('Shuffle Test')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    z = (real_rate - mean_shuffled) / (std_shuffled + 1e-9)
    print("Check 4: Shuffle Test")
    print(f"Real Phase 2 outlier rate: {real_rate:.2%}")
    print(f"Shuffle mean outlier rate: {mean_shuffled:.2%}")
    print(f"Shuffle std: {std_shuffled:.2%}")
    print(f"Z-score (real vs shuffle): {z:.2f}")
    if abs(z) < 2:
        print("Warning: Real rate is within 2 std of shuffle, result is likely artefact, not genuine signal")
    else:
        print(f"Success: Real rate is {z:.1f} std from shuffle, suggests genuine signal beyond noise")
    print()

def main():
    parser = argparse.ArgumentParser(description="SVM diagnostic checks.")
    parser.add_argument("--phase1", type=str,
                        default="assets/data/phase1_extra_features_processed.csv")
    parser.add_argument("--phase2", type=str,
                        default="assets/data/phase2_extra_features_processed.csv")
    parser.add_argument("--nu", type=float, default=0.10,
                        help="Nu for gamma sweep and shuffle test.")
    parser.add_argument("--self_report_rate", type=float, default=0.36,
                        help="Fraction of participants self-reporting stress.")
    parser.add_argument("--shuffle_repeats", type=int, default=200,
                        help="Number of shuffle iterations.")
    parser.add_argument("--variance_threshold", type=float, default=0.90,
                        help="PCA variance threshold.")
    args = parser.parse_args()

    X_train, feature_cols = load_features(args.phase1)
    X_test,  _            = load_features(args.phase2)

    # Fit PCA on Phase 1 only
    pca, X_train_pca = fit_pca(X_train, args.variance_threshold)
    X_test_pca = pca.transform(X_test)

    print(f"\nPCA: {X_train.shape[1]} features -> "
          f"{X_train_pca.shape[1]} components ({args.variance_threshold:.0%} variance)")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("One-Class SVM Diagnostic Checks", fontsize=15, y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1a = fig.add_subplot(gs[0, 0])  # PCA cumulative
    ax1b = fig.add_subplot(gs[0, 1])  # PCA individual
    ax2  = fig.add_subplot(gs[0, 2])  # Nu sweep
    ax3  = fig.add_subplot(gs[1, 0])  # Gamma sweep
    ax4  = fig.add_subplot(gs[1, 1:]) # Shuffle test (wide)

    # Run checks
    check_pca_variance(X_train, feature_cols, ax1a, ax1b)
    check_nu_sweep(X_train_pca, X_test_pca, args.self_report_rate, ax2)
    check_gamma_sweep(X_train_pca, X_test_pca, args.nu, ax3)
    shuffle_test(X_train_pca, X_test_pca, args.nu,
                       args.shuffle_repeats, ax4)

    plt.tight_layout()
    plt.savefig("svm_diagnostics.png", dpi=150, bbox_inches='tight')
    print("Diagnostic figure saved to svm_diagnostics.png")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    main()