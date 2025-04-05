# import pandas as pd
# import json

# def csv_to_rfuzz(csv_file, output_json):
#     df = pd.read_csv(csv_file, encoding="latin-1")
    
#     mutation_vectors = []
    
#     for _, row in df.iterrows():
#         vector = []
#         for val in row:
#             if isinstance(val, str):
#                 vector.extend([ord(c) for c in val][:8])  # convert string to ASCII, limited length
#             elif pd.api.types.is_numeric_dtype(type(val)):
#                 binary = format(int(val), '08b')  # integer values to 8-bit binary
#                 vector.extend([int(bit) for bit in binary])
#         mutation_vectors.append(vector[:64])  # RFUZZ typically works on fixed-length inputs
        
#     with open(output_json, "w") as json_file:
#         json.dump(mutation_vectors, json_file)
#     print(f"Successfully converted {csv_file} to {output_json}")

# # Example usage
# datasets = {
#     "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/normal_state.csv": "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/normal_rfuzz.json",
#     "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/overvoltage.csv": "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/overvoltage_rfuzz.json",
#     "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/clock_glitching.csv": "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/clock_rfuzz.json",
#     "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/rowhammer.csv": "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/rowhammer_rfuzz.json",
#     "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/unknown_data.csv": "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HJSON/unknown_rfuzz.json"
# }

# for csv_file, output_json in datasets.items():
#     csv_to_rfuzz(csv_file, output_json)



#########################################################################################################################

# import pandas as pd
# import os

# def csv_to_hwf(csv_file, hwf_file):
#     """
#     Convert each row of a CSV file into a line (instruction) in a .hwf file.
#     Adjust the format to match your harness/testbench grammar.
#     """
#     df = pd.read_csv(csv_file, encoding="latin-1")
    
#     with open(hwf_file, 'w') as f:
#         f.write("# Generated from FFAT dataset\n")
#         f.write("# CMD format: CMD V<VOLT> T<THRESH> D<DURATION> P<POWER> S<SEVERITY>\n\n")

#         for idx, row in df.iterrows():
#             try:
#                 voltage = row.get('Voltage(V)', 0)
#                 threshold = row.get('Threshold(V)', 0)
#                 duration = row.get('Duration(ms)', 0)
#                 power = row.get('Power(mW)', 0)
#                 severity = row.get('Severity(%)', 0)

#                 cmd_line = f"CMD V{voltage} T{threshold} D{duration} P{power} S{severity}"
#                 f.write(cmd_line + "\n")

#             except Exception as e:
#                 print(f"[ERROR] Row {idx}: {e}")

#     print(f"[SUCCESS] Created {hwf_file} with {len(df)} entries.")


# if __name__ == "__main__":
#     # Example usage:
#     # 1) normal_state.csv -> normal_state.hwf
#     csv_to_hwf("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/normal_state.csv", "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/normal_state.hwf")
    
#     # 2) rowhammer.csv -> rowhammer.hwf
#     csv_to_hwf("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/rowhammer.csv", "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/rowhammer.hwf")

#     csv_to_hwf("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/clock_glitching.csv", "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/clock_glitching.hwf")

#     csv_to_hwf("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/overvoltage.csv", "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/overvoltage.hwf")

#     csv_to_hwf("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_CSV/unknown_data.csv", "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FFAT_HWF/unknown_data.hwf")    





################################################################################################################################################################


# import hjson
# import csv
# import os

# def flatten_dict(d, parent_key='', sep='_'):
#     """
#     Recursively flattens a nested dictionary.
#     For example, {'a': {'b': 1}} becomes {'a_b': 1}.
#     """
#     items = {}
#     for k, v in d.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.update(flatten_dict(v, new_key, sep=sep))
#         else:
#             items[new_key] = v
#     return items

# def hjson_to_csv(hjson_path, csv_path):
#     """
#     Loads a HJSON file, flattens its contents, and writes them as one row to a CSV file.
#     """
#     if not os.path.isfile(hjson_path):
#         print(f"Error: {hjson_path} does not exist!")
#         return

#     with open(hjson_path, "r") as f:
#         data = hjson.load(f)

#     flat_data = flatten_dict(data)
    
#     # Write out the flattened data as a single CSV row
#     with open(csv_path, "w", newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=flat_data.keys())
#         writer.writeheader()
#         writer.writerow(flat_data)
    
#     print(f"Successfully converted {hjson_path} to {csv_path}")

# if __name__ == "__main__":
#     # Change these file names as needed
#     hjson_file = "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FHLS_HJSON/lock_cpp_afl_test.hjson"    # Replace with your HJSON file name
#     csv_file = "E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FHLS_CSV/lock_cpp_afl_test.csv"        # Desired CSV output file name
    
#     hjson_to_csv(hjson_file, csv_file)


##################################################################################################################################################

# import pandas as pd
# import numpy as np
# import random

# # Load original Trippel CSV
# df_orig = pd.read_csv("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FHLS_CSV/cpp_afl2.csv")

# # Number of synthetic rows to generate
# N = 500

# # Sample functions
# def synthesize_row(df):
#     row = {}
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             values = df[col].dropna().unique()
#             row[col] = random.choice(values) if len(values) > 0 else "unknown"

#         elif df[col].dtype in ['int64', 'float64']:
#             col_values = df[col].dropna()
#             if len(col_values) == 0:
#                 row[col] = 0
#             else:
#                 mean = col_values.mean()
#                 std = col_values.std()

#                 # Inject variability even if std is zero or NaN
#                 if np.isnan(std) or std == 0:
#                     std = max(1, abs(mean * 0.05))  # use 5% of mean as jitter

#                 synthetic_value = np.random.normal(loc=mean, scale=std)
#                 row[col] = max(0, int(synthetic_value))  # clip at 0 for safety

#         else:
#             row[col] = df[col].iloc[0]  # fallback
#     return row


# # Generate synthetic rows
# synthetic_rows = [synthesize_row(df_orig) for _ in range(N)]
# df_synthetic = pd.DataFrame(synthetic_rows)

# # Combine with original (optional)
# df_combined = pd.concat([df_orig, df_synthetic], ignore_index=True)

# # Save for ML model use
# df_combined.to_csv("E:/PhD Projects/FFAT/Code_FFAT/FHLS_FFAT_Comparison/FHLS_CSV/u_cpp_afl2.csv", index=False)
# print(f"[INFO] Synthetic dataset created: {df_combined.shape[0]} rows")



import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance_by_model(model, feature_names):
    """
    Plots the feature importances in descending order
    based on the model's actual numeric importances,
    but uses user-friendly names on the x-axis.
    """
    # Map each original feature name to your desired display name.
    rename_map = {
        "comp_err":          "Computation errors",
        "affected_rows":     "Affected Rows",
        "bit_flips":         "Bit flips",
        "voltage":           "Voltage",
        "timing_violations": "Timing violations",
        "duration":          "Duration",
        "severity":          "Severity",
        "power":             "Power",
        "temperature":       "Temperature",
        "timestamp":         "Timestamp",
        "status":            "Status"
    }

    # Extract importances and sort them
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]  # descending order

    # Apply the same sort to the feature names
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    # Convert to user-friendly names, if available
    friendly_names = [rename_map.get(f, f) for f in sorted_features]

    # Plot
    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(range(len(sorted_importances)), sorted_importances, color="green")
    plt.xticks(range(len(sorted_importances)), friendly_names, rotation=45)
    plt.title("Feature Importances (Sorted by Model)")
    plt.xlabel("Feature", fontweight="bold")
    plt.ylabel("Importance", fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Feature_Importances_Model_Sorted.png", dpi=300)
    plt.show()
