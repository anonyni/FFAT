import pandas as pd
import json

# Load your data (e.g., overvoltage.csv)
df = pd.read_csv("E:/PhD Projects/FFAT/Code_FFAT/RIoTfuzz_original/unknown_data.csv", encoding="latin1")
print(df.columns.tolist())

# Example transformation (for MQTT/HTTP request simulation)
payloads = []
for index, row in df.iterrows():
    payload = {
        "timestamp": row["Timestamp"],
        "voltage": row.get("Voltage(V)", row.get("Voltage", None)),
        "temperature": row.get("Temperature(°C)", row.get("Temperature", None)),
        "power": row.get("Power(mW)", row.get("Power", None)),
        "severity": row.get("Severity(%)", row.get("Severity", None)),
        "duration": row.get("Duration(ms)", row.get("Duration", None)),
        "attack_type": "Overvoltage"
    }
    payloads.append(payload)

# Save as JSON
with open("E:/PhD Projects/FFAT/Code_FFAT/RIoTfuzz_original/unknown_data.json", "w") as file:
    json.dump(payloads, file, indent=4)
