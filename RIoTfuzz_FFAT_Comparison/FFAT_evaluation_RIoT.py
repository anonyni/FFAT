import json
import os
from collections import Counter

# Define JSON dataset filenames
datasets = [
    "E:/PhD Projects/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/clock_glitching.json",
    "E:/PhD Projects/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/normal_state.json",
    "E:/PhD Projects/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/overvoltage.json",
    "E:/PhD Projects/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/rowhammer.json",
    "E:/PhD Projects/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/unknown_data.json"
]

# Helper function to determine the data type clearly
def determine_type(value):
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    else:
        return "string"

# Unified evaluation results storage
summary_results = {}

# Iterate through all provided datasets
for dataset_filename in datasets:
    if not os.path.exists(dataset_filename):
        print(f"Dataset {dataset_filename} not found.")
        continue
    
    # Load dataset JSON entries
    with open(dataset_filename, 'r') as json_file:
        dataset_entries = json.load(json_file)
    
    data_type_counter = Counter()
    slow_response_entries = 0
    
    # Process each entry in the JSON dataset
    for entry in dataset_entries:
        for key, value in entry.items():
            # Update data type counter
            value_type = determine_type(value)
            data_type_counter[value_type] += 1
        
        # Evaluate the response time if applicable (assuming 'duration' in ms)
        if 'duration' in entry:
            response_duration_sec = entry['duration'] / 1000.0  # Convert to seconds
            if response_duration_sec > 0.3:
                slow_response_entries += 1
    
    # Store summary of each dataset
    summary_results[dataset_filename] = {
        'total_entries': len(dataset_entries),
        'data_type_distribution': dict(data_type_counter),
        'slow_response_count': slow_response_entries
    }

# Output structured summary clearly
print("\n========== Unified Datasets Evaluation Summary ==========\n")

for dataset, results in summary_results.items():
    print(f"Dataset: {dataset}")
    print(f"  - Total Entries: {results['total_entries']}")
    print(f"  - Slow Response (>0.3s): {results['slow_response_count']}")
    print("  - Data Type Counts:")
    for dtype, count in results['data_type_distribution'].items():
        print(f"      * {dtype}: {count}")
    print("-------------------------------------------------------")
