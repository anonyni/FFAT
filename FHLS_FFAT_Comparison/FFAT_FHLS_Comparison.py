# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # Raw results
# datasets = ['Normal State', 'Overvoltage', 'Clock Glitching', 'Rowhammer', 'Unknown']

# # FFAT model performance (binary accuracy, then multiclass accuracy where applicable)
# ffat_accuracy = [0.77, 1.00, 1.00, 1.00, 1.00]  # You can adjust with exact values if needed

# # RFUZZ metrics from your output
# rfuzz_coverage = [25.81, 30.36, 766.40, 28.99, 14.20]
# rfuzz_faults = [0, 272, 585, 152, 0]

# # Normalize all metrics to [0,1] for visual comparison
# def normalize(x):
#     x = np.array(x, dtype=float)
#     return (x - x.min()) / (x.max() - x.min())

# data = {
#     'FFAT Accuracy': normalize(ffat_accuracy),
#     'RFUZZ Coverage': normalize(rfuzz_coverage),
#     'RFUZZ Faults Detected': normalize(rfuzz_faults)
# }

# df = pd.DataFrame(data, index=datasets)

# # Create a heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Normalized Score'})
# plt.title("FFAT vs RFUZZ Comparison on FFAT Datasets", fontsize=14)
# plt.xlabel("Evaluation Metric")
# plt.ylabel("Dataset")
# plt.tight_layout()
# plt.savefig("FFAT_vs_RFUZZ_heatmap.pdf", dpi=300)
# plt.show()

###############################################################################################################################

# import matplotlib.pyplot as plt
# import numpy as np

# # Sample data (replace with your actual numbers)
# datasets = ['Glitching', 'Rowhammer', 'Overvoltage', 'Unknown1']
# trippel_coverage = [25.81, 30.36, 766.40, 14.20]
# trippel_faults = [0, 272, 585, 0]
# ffat_accuracy = [98.5, 96.4, 97.9, 85.1]

# x = np.arange(len(datasets))  # the label locations
# width = 0.25  # the width of the bars

# fig, ax = plt.subplots(figsize=(10, 6))
# rects1 = ax.bar(x - width, trippel_coverage, width, label='Trippel Coverage (%)')
# rects2 = ax.bar(x, trippel_faults, width, label='Trippel Faults')
# rects3 = ax.bar(x + width, ffat_accuracy, width, label='FFAT Accuracy (%)')

# # Add labels, title, legend
# ax.set_ylabel('Value', fontweight="bold")
# ax.set_title('Comparison: Trippel Hardware Fuzzing Evaluation vs. FFAT')
# ax.set_xticks(x)
# ax.set_xticklabels(datasets)
# ax.legend()

# # Label bars
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.1f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 5),  # offset
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=8)

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

# plt.tight_layout()
# plt.savefig("comparison_chart.png")
# plt.show()


#########################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a dictionary where each key is a feature and each value is a list of its importance per test.
# If a test does not have a value for that feature, we use np.nan.
data = {
    "fuzzer_params_timeout_ms": [0.0656,      np.nan,       np.nan,       np.nan],
    "tb":                        [0.0634,      np.nan,       0.0633,       np.nan],
    "fuzzer_params_num_instances":[0.0616,    0.0552,       np.nan,       np.nan],
    "hdl_gen_params_num_lock_states": [0.0614, np.nan,       np.nan,       np.nan],
    "hdl_gen_params_lock_comp_width": [0.0613, np.nan,       np.nan,       np.nan],
    "fuzzer_params_mode":        [np.nan,     0.0570,       np.nan,       0.0564],
    "soc":                       [np.nan,     0.0552,       np.nan,       np.nan],
    "fuzzer":                    [np.nan,     0.0551,       np.nan,       np.nan],
    "default_input":             [np.nan,     0.0516,       np.nan,       0.0538],
    "tb_type":                   [np.nan,     np.nan,       0.0625,       np.nan],
    "instrument_dut":            [np.nan,     np.nan,       0.0583,       0.0561],
    "manual":                    [np.nan,     np.nan,       0.0544,       np.nan],
    "fuzzer_params_memory_limit_mb": [np.nan, np.nan,       0.0542,       0.0551],
    "fuzzer_params_interactive_mode": [np.nan, np.nan,       np.nan,       0.0511]
}

# Define the test names as the index of the DataFrame.
tests = [
    "lock_cpp_afl_test",
    "aes_test_template",
    "cpp_afl2",
    "lock_cocotb_afl_test"
]

df = pd.DataFrame(data, index=tests)

# Print the DataFrame (optional)
print(df)

# Plotting the heatmap using matplotlib:
plt.figure(figsize=(12, 6))

# Create a masked array so that NaN values are not plotted.
masked_array = np.ma.masked_invalid(df.values)

# Display the heatmap (using the default colormap)
cmap = plt.cm.get_cmap('coolwarm')
heatmap = plt.imshow(masked_array, aspect='auto', interpolation='none', cmap=cmap)
plt.colorbar(heatmap)
# Set ticks and labels for both axes.
plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, rotation=90)
plt.yticks(ticks=np.arange(len(df.index)), labels=df.index)

# Optionally, annotate each cell with the numeric value.
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        value = df.values[i, j]
        if not np.isnan(value):
            plt.text(j, i, f'{value:.4f}', ha='center', va='center')

plt.title("Important Features Across Tests", fontweight = "bold")
plt.tight_layout()
plt.show()



# Performance metrics per test
metrics = {
    "lock_cpp_afl_test": {"accuracy": 0.7067, "precision_1": 0.71, "recall_1": 0.97, "f1_1": 0.82},
    "aes_test_template": {"accuracy": 0.7152, "precision_1": 0.73, "recall_1": 0.97, "f1_1": 0.83},
    "cpp_afl2": {"accuracy": 0.6533, "precision_1": 0.68, "recall_1": 0.94, "f1_1": 0.79},
    "lock_cocotb_afl_test": {"accuracy": 0.66, "precision_1": 0.68, "recall_1": 0.96, "f1_1": 0.79},
}

# # Top features per test
# features = {
#     "lock_cpp_afl_test": {"tb": 0.0634},
#     "aes_test_template": {"soc": 0.0552},
#     "cpp_afl2": {"tb": 0.0633},
#     "lock_cocotb_afl_test": {}
# }

# Merge into one big dict
combined = {}
all_columns = set()
for test in metrics:
    combined[test] = {}
    combined[test].update(metrics[test])
    # combined[test].update(features.get(test, {}))
    all_columns.update(combined[test].keys())

# Convert to DataFrame
df_combined = pd.DataFrame.from_dict(combined, orient="index")[sorted(all_columns)]

# Plot heatmap
plt.figure(figsize=(14, 6))
masked = np.ma.masked_invalid(df_combined.values.astype(float))
cmap = plt.cm.plasma.copy()
cmap.set_bad(color='lightgrey')

cmap = plt.cm.get_cmap('coolwarm')
heatmap = plt.imshow(masked, aspect='auto', interpolation='none', cmap=cmap)
plt.colorbar(heatmap)

plt.xticks(ticks=np.arange(len(df_combined.columns)), labels=df_combined.columns, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(df_combined.index)), labels=df_combined.index)

for i in range(masked.shape[0]):
    for j in range(masked.shape[1]):
        val = masked[i, j]
        if not np.ma.is_masked(val):
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=10, color='white' if val > 0.9 else 'black')

plt.title("Performance Metrics", fontsize=12, fontweight = "bold")
plt.tight_layout()
plt.show()
















