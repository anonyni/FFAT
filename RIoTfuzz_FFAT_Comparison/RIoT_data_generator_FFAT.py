import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Function to generate random timestamps
def generate_timestamps(start, num_entries, interval_seconds=1):
    timestamps = [start + timedelta(seconds=i * interval_seconds) for i in range(num_entries)]
    return timestamps

# Generate simulated fuzz-testing dataset
def generate_simulated_riotfuzz_data(filename, num_entries, attack_label):
    timestamps = generate_timestamps(datetime.now(), num_entries)
    parameters = ['Voltage', 'Current', 'Frequency', 'Power', 'Temperature']
    results = ['success', 'fail', 'timeout', 'error']
    
    data = []
    for ts in timestamps:
        param = random.choice(parameters)
        value = round(np.random.uniform(0, 5), 3) if param == 'Voltage' else \
                round(np.random.uniform(0, 10), 3) if param == 'Current' else \
                round(np.random.uniform(1e3, 1e6), 1) if param == 'Frequency' else \
                round(np.random.uniform(0, 100), 2) if param == 'Power' else \
                round(np.random.uniform(20, 80), 2)  # Temperature
        
        response_time = round(np.random.uniform(0.05, 0.6), 3)  # Simulate response time between 50ms and 600ms
        
        # Add higher response times and error rates for attacks
        if attack_label != 'normal_state':
            response_time *= random.uniform(1.2, 2.5)
            result = random.choices(results, weights=[60, 20, 10, 10])[0]
        else:
            result = random.choices(results, weights=[95, 3, 1, 1])[0]
        
        data.append({
            'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'Parameter': param,
            'Value': value,
            'Response_Time': response_time,
            'Result': result,
            'Label': attack_label
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {num_entries} entries.")

# Generating simulated datasets:
generate_simulated_riotfuzz_data("E:/FFAT/Code_FFAT/RIoTfuzz_original/RIoT_CSV/riotfuzz_normal.csv", 1000, "normal_state")
generate_simulated_riotfuzz_data("E:/FFAT/Code_FFAT/RIoTfuzz_original/RIoT_CSV/riotfuzz_mutation_attack.csv", 1000, "mutation_attack")
