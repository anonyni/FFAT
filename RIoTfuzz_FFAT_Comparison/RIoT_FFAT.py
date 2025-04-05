# compare_models.py

"""
This script performs a full cross-comparison between:
1. Your ML classifier applied on your CSV dataset.
2. A simple RIoTFuzzer-style inference simulation (based on response time threshold)
   applied on the same dataset.
It exports evaluation metrics and comparative visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ----------------------
# CONFIGURATION
# ----------------------
YOUR_CSV_FILES = {
    "normal_state": r"C:/Users/umara/OneDrive - National University of Singapore/Desktop/RIoTfuzz_original/normal_state.csv",
    "overvoltage": r"C:/Users/umara/OneDrive - National University of Singapore/Desktop/RIoTfuzz_original/overvoltage.csv",
    "rowhammer": r"C:/Users/umara/OneDrive - National University of Singapore/Desktop/RIoTfuzz_original/rowhammer.csv",
    "clock_glitching": r"C:/Users/umara/OneDrive - National University of Singapore/Desktop/RIoTfuzz_original/clock_glitching.csv",
    "unknown": r"C:/Users/umara/OneDrive - National University of Singapore/Desktop/RIoTfuzz_original/unknown_data.csv"
}
RESULTS_DIR = "comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------
# LOAD + PREPROCESS
# ----------------------
def load_your_dataset():
    dfs = []
    for label, path in YOUR_CSV_FILES.items():
        df = pd.read_csv(path, encoding='latin-1')
        df['Label'] = label
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def preprocess(df):
    # Optionally drop timestamp or other non-feature columns
    df = df.drop(columns=['Timestamp'], errors='ignore')
    label_encoder = LabelEncoder()
    df['Label_encoded'] = label_encoder.fit_transform(df['Label'])
    
    # Identify non-numeric columns (excluding label columns)
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in ['Label', 'Label_encoded']:
        if col in non_numeric_cols:
            non_numeric_cols.remove(col)
    
    # Option 1: Drop non-numeric columns
    # df_numeric = df.drop(columns=non_numeric_cols, errors='ignore')
    
    # Option 2 (if you want to keep some non-numeric columns, encode them instead)
    for col in non_numeric_cols:
        df[col] = LabelEncoder().fit_transform(df[col].str.strip())
    df_numeric = df
    
    X = df_numeric.drop(columns=['Label', 'Label_encoded'], errors='ignore')
    y = df['Label_encoded']
    
    # Impute missing values using the mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, y, label_encoder, scaler
# ----------------------
# MODEL TRAINING
# ----------------------
def train_classifier(X, y, name="YourClassifier"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = clf.score(X_test, y_test)
    logging.info(f"{name} Accuracy: {acc:.2f}")
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=np.unique(y))
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"{RESULTS_DIR}/{name}_confusion_matrix.png")
    plt.close()
    return clf, report

# ----------------------
# RIoTFuzzer-style Inference Simulation
# ----------------------
def simulate_riotfuzzer_inference(df):
    # For demonstration, we simulate RIoTFuzzer by using a threshold on a 'Duration(ms)' column.
    # In real RIoTFuzzer, side-channel response times (e.g., >300ms) are used to infer a successful packet.
    if 'Duration(ms)' not in df.columns:
        logging.warning("Duration(ms) column missing, skipping RIoTFuzzer simulation.")
        return None
    threshold = 300  # ms threshold
    # Heuristically mark non-normal states as anomalies
    predicted = (df['Duration(ms)'] > threshold).astype(int)
    # For binary comparison, treat 'normal_state' as 0 and any other label as 1.
    actual = (df['Label'] != 'normal_state').astype(int)
    report = classification_report(actual, predicted, output_dict=True)
    logging.info("RIoTFuzzer-style inference simulation completed.")
    # Plot confusion matrix
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("RIoTFuzzer Inference Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{RESULTS_DIR}/RIoTFuzzer_inference_confusion_matrix.png")
    plt.close()
    return report

# ----------------------
# PCA Visualization
# ----------------------
def pca_visualization(X, y, title="PCA Visualization"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=y, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter)
    plt.savefig(f"{RESULTS_DIR}/{title.replace(' ', '_')}.png")
    plt.close()

# ----------------------
# MAIN COMPARISON LOGIC
# ----------------------
def main():
    df = load_your_dataset()
    X, y, le, scaler = preprocess(df)
    
    # Train your classifier
    clf, report_classifier = train_classifier(X, y, name="YourClassifierOnYourData")
    
    # Simulate RIoTFuzzer inference on your dataset
    report_riotf = simulate_riotfuzzer_inference(df.copy())
    
    # PCA visualization
    pca_visualization(X, y, title="Your Data PCA")
    
    # Save reports as CSV files
    classifier_report_path = "your_classifier_report.csv"
    pd.DataFrame(report_classifier).T.to_csv(classifier_report_path)
    print(f"Classifier report saved to {classifier_report_path}")
    
    if report_riotf is not None:
        riotf_report_path = "riotfuzzer_inference_report.csv"
        pd.DataFrame(report_riotf).T.to_csv(riotf_report_path)
        print(f"RIoTFuzzer inference report saved to {riotf_report_path}")

if __name__ == "__main__":
    main()

