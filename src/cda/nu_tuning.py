import numpy as np
import matplotlib.pyplot as plt
from SVM import run_global_anomaly_detection

phase1_path = "assets/data/phase1_processed.csv"
phase2_path = "assets/data/phase2_processed.csv"


nu_values = np.linspace(0.01, 0.30, 20)
test_outlier_ratios = []

for nu in nu_values:
    # Use your existing run_global_anomaly_detection function
    stats = run_global_anomaly_detection(phase1_path, phase2_path, nu=nu, plot=False)
    test_outlier_ratios.append(stats['Test_Outlier_Ratio'])

plt.plot(nu_values, test_outlier_ratios, marker='o')
plt.xlabel('nu parameter')
plt.ylabel('Phase 2 Outlier Ratio')
plt.title('OC-SVM Sensitivity to nu')
plt.grid(True)
plt.show()