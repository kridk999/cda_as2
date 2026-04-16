# load data from csv file and detect missing values
import pandas as pd
import glob

def detect_missing_values(file_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Detect missing values
        missing_values = df.isnull().sum()

        # Only print if there are actually missing values to keep the output clean
        if missing_values.sum() > 0:
            print(f"\nMissing values in {file_path}:")
            print(missing_values[missing_values > 0])
            
    except Exception as e:
        print(f"\nError reading {file_path}: {e}")

base_path = 'assets/dataset/'

# Use glob to recursively find all .csv files within the nested folder structure
# ** will match all directories and subdirectories
csv_files = glob.glob(f"{base_path}/**/*.csv", recursive=True)

print(f"Found {len(csv_files)} CSV files. Checking for missing values...\n")

for file in csv_files:
    detect_missing_values(file)
