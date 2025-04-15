import json
import urllib.parse
import os

###########################################
# Global Configuration (Based on Original)
###########################################

# Updated mappings
attack_type_to_siid = {
    "Overvoltage": 2,
    "Clock Glitching": 3,
    "Rowhammer": 5,
    "Normal": 1,
    "Unknown": 4
}

piid_mapping = {
    "temperature": 1,
    "power": 2,
    "severity": 3,
    "duration": 4,
    "voltage": 5
}

data_type_list = ["float", "string", "int", "boolean"]
key_2_list = [1, 2, 3, 4, 5]

control_commands = [{
    1: [1, 2, 3, 4, 5],
    2: [1, 2, 3, 4, 5],
    3: [1, 2, 3, 4, 5],
    4: [1, 2, 3, 4, 5],
    5: [1, 2, 3, 4, 5]
}]

def initialize_data_type_index():
    """Initialize the 4D array based on updated control_commands."""
    return [
        [
            [[0 for _ in range(4)] for _ in range(len(control_commands[0][siid]))]
            for siid in key_2_list
        ]
    ]

def determine_type(value):
    if value is None:
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    try:
        float(value)
        return "float"
    except:
        pass
    return "string"

def process_fuzzing_dataset_1(json_filename):
    if not os.path.exists(json_filename):
        print(f"File {json_filename} not found.")
        return

    with open(json_filename, "r") as f:
        entries = json.load(f)

    print(f"\nProcessing dataset: {json_filename} with {len(entries)} entries")

    local_dt_index = initialize_data_type_index()

    for idx, entry in enumerate(entries):
        attack_type = entry.get("attack_type")
        if attack_type not in attack_type_to_siid:
            print(f"Unknown attack type '{attack_type}', skipping entry.")
            continue

        key_2 = attack_type_to_siid[attack_type]
        key_index_2 = key_2_list.index(key_2)

        for param, piid in piid_mapping.items():
            value = entry.get(param, None)
            value_type = determine_type(value)
            dtype_index = data_type_list.index(value_type)

            # Identify the command index for piid
            if piid in control_commands[0][key_2]:
                j = control_commands[0][key_2].index(piid)

                # Increment counters based on response time condition
                response_time = entry.get("duration", 200) / 1000.0  # Default duration 0.2s if not provided
                if local_dt_index[0][key_index_2][j][dtype_index] == 10:
                    continue  # already reached max count
                elif response_time <= 0.3:
                    local_dt_index[0][key_index_2][j][dtype_index] += 1
                else:
                    local_dt_index[0][key_index_2][j][dtype_index] = -999

    print(f"\nFinal data_type_index_list from {json_filename}:\n{local_dt_index}")

    for i in range(len(local_dt_index)):
        for j in range(len(local_dt_index[i])):
            for k in range(len(local_dt_index[i][j])):
                for dtype_index, count in enumerate(local_dt_index[i][j][k]):
                    if count == 10:
                        key_for_cmd = key_2_list[j]
                        command = control_commands[i][key_for_cmd][k]
                        print(f"Reached 10 counts for siid {key_for_cmd}, piid {command}, data type {data_type_list[dtype_index]}.")

##########################################################################
# Main entry point: Run both modified evaluations
##########################################################################
##############################################
# Main Execution: Process Each JSON Dataset
##############################################
if __name__ == "__main__":
    # List your JSON dataset files.
    dataset_files = [
    "E:/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/clock_glitching.json",
    "E:/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/normal_state.json",
    "E:/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/overvoltage.json",
    "E:/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/rowhammer.json",
    "E:/FFAT/Code_FFAT/RIoTfuzz_original/FFAT_JSON/unknown_data.json"
    ]
    
    # Process each file with the first evaluation function.
    for ds in dataset_files:
        print("\n==============================")
        process_fuzzing_dataset_1(ds)
        # Uncomment the next line if you wish to also run evaluation version 2.
        # process_fuzzing_dataset_2(ds)
