import os
import csv
import random
import datetime
import pandas as pd

# Configuration for the dataset structure
BASE_DIR = os.path.join("assets", "dummy_dataset")
NUM_COHORTS = 6       # D1_1 to D1_6
NUM_IDS = 3           # IDs per cohort
NUM_ROUNDS = 4        # round_1 to round_4
NUM_PHASES = 3        # phase1 to phase3
VECTOR_LENGTH = 10    # Number of simulated data points per vector

def format_timestamp(dt, file_type):
    """Format the datetime object to match specific file string representations."""
    if file_type == "HR":
        # e.g., 2021-10-17 13:11:55.
        return dt.strftime('%Y-%m-%d %H:%M:%S.')
    elif file_type == "BVP":
        # e.g., 2021-12-14 10:11:54.953125
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    else:  # EDA, TEMP
        # e.g., 2024-11-17 18:11:55.000
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def generate_dummy_dataset():
    # Base starting date for simulation
    start_date = datetime.datetime(2021, 10, 17, 13, 11, 55)
    
    # Iterate through the defined structure
    for cohort_idx in range(1, NUM_COHORTS + 1):
        cohort_name = f"D1_{cohort_idx}"
        
        for id_idx in range(1, NUM_IDS + 1):
            id_name = f"ID_{id_idx}"
            
            for round_idx in range(1, NUM_ROUNDS + 1):
                round_name = f"round_{round_idx}"
                
                for phase_idx in range(1, NUM_PHASES + 1):
                    phase_name = f"phase{phase_idx}"
                    
                    dir_path = os.path.join(BASE_DIR, cohort_name, id_name, round_name, phase_name)
                    os.makedirs(dir_path, exist_ok=True)
                    
                    # Generate a timestamp specific to this phase
                    phase_time = start_date + datetime.timedelta(days=random.randint(1, 100), hours=random.randint(1, 12))
                    
                    # File definitions: [Value column name, generating function, time step delta]
                    # BVP is sampled at 64Hz typically (~15.6ms), HR at 1Hz, EDA and TEMP at 4Hz
                    files = {
                        "HR.csv": {"col": "HR", "val": lambda: round(random.uniform(60, 100), 2), "td": datetime.timedelta(seconds=1)},
                        "EDA.csv": {"col": "EDA", "val": lambda: round(random.uniform(0.1, 5.0), 6), "td": datetime.timedelta(milliseconds=250)},
                        "BVP.csv": {"col": "BVP", "val": lambda: round(random.uniform(-100, 100), 2), "td": datetime.timedelta(microseconds=15625)},
                        "TEMP.csv": {"col": "TEMP", "val": lambda: round(random.uniform(30.0, 36.0), 2), "td": datetime.timedelta(milliseconds=250)}
                    }
                    
                    for filename, info in files.items():
                        file_type = filename.split('.')[0]
                        file_path = os.path.join(dir_path, filename)
                        
                        with open(file_path, mode='w', newline='') as f:
                            writer = csv.writer(f)
                            # Empty string for the index column
                            writer.writerow(["", info["col"], "time"])
                            
                            current_time = phase_time
                            # Generate 10 rows of dummy data for each file
                            for row_idx in range(10):
                                val = info["val"]()
                                time_str = format_timestamp(current_time, file_type)
                                
                                writer.writerow([row_idx, f"{val:.2f}" if file_type != "EDA" else f"{val:.6f}", time_str])
                                
                                # increment time
                                current_time += info["td"]

def generate_dummy_dataframes():
    dataframes = {}

    for phase_idx in range(1, NUM_PHASES + 1):
        phase_name = f"Phase_{phase_idx}"
        rows = []
        
        for cohort_idx in range(1, NUM_COHORTS + 1):
            cohort_name = f"C{cohort_idx}"
            
            for id_idx in range(1, NUM_IDS + 1):
                id_name = f"ID{id_idx}"
                
                for round_idx in range(1, NUM_ROUNDS + 1):
                    round_name = f"R{round_idx}"
                    
                    # Generate data vectors (lists of floats)
                    hr_vector = [round(random.uniform(60, 100), 2) for _ in range(VECTOR_LENGTH)]
                    eda_vector = [round(random.uniform(0.1, 5.0), 6) for _ in range(VECTOR_LENGTH)]
                    bvp_vector = [round(random.uniform(-100, 100), 2) for _ in range(VECTOR_LENGTH)]
                    temp_vector = [round(random.uniform(30.0, 36.0), 2) for _ in range(VECTOR_LENGTH)]
                    
                    rows.append({
                        "Cohort": cohort_name,
                        "Person": id_name,
                        "Round": round_name,
                        "HR_vector": hr_vector,
                        "EDA_vector": eda_vector,
                        "BVP_vector": bvp_vector,
                        "TEMP_vector": temp_vector
                    })
        
        # Create DataFrame for the current phase
        df = pd.DataFrame(rows)
        dataframes[phase_name] = df

    return dataframes

if __name__ == "__main__":
    generate_dummy_dataset()
    print(f"Successfully generated dataset at: {os.path.abspath(BASE_DIR)}")
    
    dfs = generate_dummy_dataframes()
    
    # Access the individual dataframes
    df_phase1 = dfs["Phase_1"]
    df_phase2 = dfs["Phase_2"]
    df_phase3 = dfs["Phase_3"]
    
    print("--- Phase 1 DataFrame Head ---")
    print(df_phase1.head())
    
    # TIP: If you want to save these DataFrames to disk and preserve the arrays/lists,
    # it is highly recommended to use Pickle or Parquet instead of CSV. 
    # CSV will convert the lists into plain strings like "[86.7, 92.1, ...]".