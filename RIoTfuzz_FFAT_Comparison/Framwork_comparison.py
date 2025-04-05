import matplotlib.pyplot as plt
import seaborn as sns

# Data format: {dataset: (FFAT_accuracy, RIoTfuzz_learned_flag)}
results = {
    "Normal": (100, 1),
    "Overvoltage": (100, 1),
    "Clock-Glitching": (98, 0),
    "Rowhammer": (100, 0),
    "Unknown": (95, 0),
    "RIoTfuzz (Mutation Attack)": (100, None),  # FFAT only
}

datasets = list(results.keys())
ffat_scores = [v[0] for v in results.values()]
riot_learned = [v[1] for v in results.values()]

x = range(len(datasets))

plt.figure(figsize=(10, 6))
plt.bar(x, ffat_scores, width=0.4, label="FFAT Accuracy", align='center')
plt.bar([i + 0.4 for i in x], [v*100 if v is not None else 0 for v in riot_learned], width=0.4,
        label="RIoTfuzz Learnability", align='center', alpha=0.7)

plt.xticks([i + 0.2 for i in x], datasets, rotation=45)
plt.ylabel("Score (%)")
plt.title("Performance Comparison: FFAT vs. RIoTfuzz")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_accuracy.png")
plt.show()


import pandas as pd

comparison_data = {
    "Dataset": ["Normal", "Clock-Glitching", "Overvoltage", "Rowhammer", "Unknown", "Mutation Attack (RIoT)"],
    "FFAT Accuracy (%)": [100, 98, 100, 100, 95, 100],
    "RIoTfuzz Learned": ["✓", "✗", "✓", "✗", "✗", "-"],
    "RIoTfuzz Slow Resps": [0, 243, 0, 570, 16, "-"]
}

df = pd.DataFrame(comparison_data)
print(df.to_markdown(index=False))  # or .to_latex()


import numpy as np

datasets = ["Normal", "Clock-Glitching", "Rowhammer", "Overvoltage", "Unknown"]
slow_responses = [0, 243, 570, 0, 16]
total = [640, 585, 570, 591, 20]

percent_slow = [round(s/t * 100, 2) for s, t in zip(slow_responses, total)]

plt.figure(figsize=(8, 5))
plt.plot(datasets, percent_slow, marker='o', color='red')
plt.ylabel("Slow Response Rate (%)")
plt.title("RIoTfuzz: Impact of Slow Responses on FFAT Dataset")
plt.grid(True)
plt.savefig("riot_slow_response.png")
plt.show()


