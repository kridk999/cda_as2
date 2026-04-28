import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

df1 = pd.read_csv('assets/data/phase1_processed.csv')
df1['Phase'] = 'Phase 1 (Pre-Puzzle Rest)'

df3 = pd.read_csv('assets/data/phase3_processed.csv')
df3['Phase'] = 'Phase 3 (Post-Puzzle Rest)'


df_compare = pd.concat([df1, df3], ignore_index=True)
features = ['BVP_standardized', 'EDA_standardized', 'HR_standardized', 'TEMP_standardized']

results = []
try:
    for feature in features:
            stat, pval = stats.ttest_ind(df1[feature], df3[feature], equal_var=False)
            results.append({
                'Feature': feature,
                'Raw P-value': pval
            })
            
except Exception as e:
    features = ['BVP_mean', 'EDA_mean', 'HR_mean', 'TEMP_mean']
    for feature in features:
        stat, pval = stats.ttest_ind(df1[feature], df3[feature], equal_var=False)
        results.append({
            'Feature': feature,
            'Raw P-value': pval
        })
results_df = pd.DataFrame(results)


reject, pvals_corr, _, _ = multipletests(results_df['Raw P-value'], alpha=0.05, method='fdr_bh')
results_df['Corrected P-value (FDR)'] = pvals_corr
results_df['Reject Null (Sig. Difference)'] = reject


print(results_df.to_string(index=False))


sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
colors = ['#4C72B0', '#C44E52'] 
for idx, feature in enumerate(features):
    sns.violinplot(
        x='Phase', y=feature, hue='Phase', data=df_compare, 
        ax=axes[idx], palette=colors, inner=None, alpha=0.4, legend=False)
    sns.boxplot(
        x='Phase', y=feature, hue='Phase', data=df_compare, 
        ax=axes[idx], palette=colors, width=0.15, boxprops={'zorder': 2}, 
        showfliers=False, legend=False)
    sns.stripplot(
        x='Phase', y=feature, hue='Phase', data=df_compare, 
        ax=axes[idx], color='black', alpha=0.4, jitter=True, size=5, legend=False)
    clean_name = feature.replace("_standardized", " (Standardized)")
    axes[idx].set_title(f'{clean_name}', fontweight='bold')
    axes[idx].set_ylabel('')
    axes[idx].set_xlabel('')
plt.suptitle('Physiological Divergence: Pre-Rest vs Post-Rest\n(Notice the extreme elevated data points in Phase 3 EDA)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
