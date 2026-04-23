import argparse
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


from data_preprocess import build_dataframe

def compare_stds(df1, df2, feature, phase_a_name, phase_b_name):
    """
    Compares standard deviations between two phases for one feature.
    Uses Brown-Forsythe (Levene with median) to test equality of variances.
    """
    data1 = df1[feature].dropna()
    data2 = df2[feature].dropna()

    std1 = data1.std(ddof=1)
    std2 = data2.std(ddof=1)

    # Robust test for equal variances
    bf_stat, bf_p = stats.levene(data1, data2, center="median")

    ratio = (std1 / std2) if std2 != 0 else float("inf")

    print(f"STD Compare - Feature: {feature}")
    print(f"{phase_a_name.capitalize()} Std: {std1:.4f} | {phase_b_name.capitalize()} Std: {std2:.4f}")
    print(f"Std Ratio ({phase_a_name}/{phase_b_name}): {ratio:.4f}")
    print(f"Brown-Forsythe Test: stat={bf_stat:.4f}, p-value={bf_p:.4e}\n")

    return std1, std2, bf_p

def compare_distributions(df1, df2, feature, phase_a_name, phase_b_name):
    """
    the Two-Sample Kolmogorov-Smirnov test and Welch's t-test.
    """
    data1 = df1[feature].dropna()
    data2 = df2[feature].dropna()
    

    ks_stat, ks_p = stats.ks_2samp(data1, data2)
    

    t_stat, t_p = stats.ttest_ind(data1, data2, equal_var=False)
    
    print(f"--- Feature: {feature} ---")
    print(f"{phase_a_name.capitalize()} Mean: {data1.mean():.4f} | {phase_b_name.capitalize()} Mean: {data2.mean():.4f}")
    print(f"KS Test: stat={ks_stat:.4f}, p-value={ks_p:.4e}")
    print(f"T-Test:  stat={t_stat:.4f}, p-value={t_p:.4e}\n")
    
    return ks_p, t_p


def plot_distributions(df_a, df_b, features, phase_a_name, phase_b_name, output_img_path):
    """
    Plots Kernel Density Estimates (KDE) to visually inspect if the 
    distributions match.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        
        df_a[feature].dropna().plot.kde(ax=ax, label=phase_a_name, color='blue', alpha=0.7)
        df_b[feature].dropna().plot.kde(ax=ax, label=phase_b_name, color='orange', alpha=0.7)
        
        ax.set_title(f"Distribution of {feature}")
        ax.legend()
        
    plt.tight_layout()
    #plt.savefig(output_img_path)
    #print(f"Saved distribution plot to {output_img_path}")
    plt.show()


def run_statistical_comparison(base_path, phase_a_name, phase_b_name, output_img_path):
    """
    Loads data for two phases from the base dataset dictionary, computes 
    statistical differences, and creates distribution plots.
    """

    df_phase_a = build_dataframe(base_path, phase_a_name)
    df_phase_b = build_dataframe(base_path, phase_b_name)
    
    features_to_compare = ["BVP_mean", "EDA_mean", "HR_mean", "TEMP_mean"]
    
    
    for feature in features_to_compare:
        compare_distributions(df_phase_a, df_phase_b, feature, phase_a_name, phase_b_name)
        compare_stds(df_phase_a, df_phase_b, feature, phase_a_name, phase_b_name)
        
    # Visual comparison
    plot_distributions(df_phase_a, df_phase_b, features_to_compare, phase_a_name, phase_b_name, output_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare statistical distributions of two data phases.")
    
    parser.add_argument("--base_path", type=str, default="assets/data/dataset", 
                        help="Path to the main dataset folder containing phase subfolders.")
    parser.add_argument("--phase_a", type=str, default="phase1", 
                        help="Name of the first phase to compare (e.g., 'phase1').")
    parser.add_argument("--phase_b", type=str, default="phase3", 
                        help="Name of the second phase to compare (e.g., 'phase3').")
    parser.add_argument("--output_img", type=str, default="assets/data/phase_comparison.png", 
                        help="Path where the generated distribution plot will be saved.")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_path)

    run_statistical_comparison(
        base_path=base_dir, 
        phase_a_name=args.phase_a, 
        phase_b_name=args.phase_b,
        output_img_path=args.output_img
    )