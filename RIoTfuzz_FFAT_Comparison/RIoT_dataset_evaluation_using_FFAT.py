import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import logging
from joblib import dump
from sklearn.decomposition import PCA
import time
import pynvml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# DATASETS dictionary (adjust file paths as needed)
DATASETS = {
    "mutation_attack": "E:/FFAT/Code_FFAT/RIoTfuzz_FFAT_Comparison/RIoT_CSV/riotfuzz_mutation_attack.csv",
    "normal_state": "E:/FFAT/Code_FFAT/RIoTfuzz_FFAT_Comparison/RIoT_CSV/riotfuzz_normal.csv"
}

# Extra file for unknown data (optional)
UNKNOWN_DATA_PATH = "attack_data/unknown_data.csv"

def load_and_combine_datasets(dataset_paths, encoding="latin-1"):
    dfs = []
    for label, path in dataset_paths.items():
        df = pd.read_csv(path, encoding=encoding)
        df["Label"] = label
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logging.info(f"Combined dataset shape: {combined.shape}")
    return combined

def load_unknown_dataset(unknown_path, encoding="latin-1"):
    df = pd.read_csv(unknown_path, encoding=encoding)
    df["Label"] = "unknown"
    logging.info(f"Unknown dataset shape: {df.shape}")
    return df

def preprocess_data(data):
    # Drop Timestamp if present
    data = data.drop(columns=["Timestamp"], errors="ignore")
    label_encoders = {}
    # Convert object columns to string to ensure uniformity
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].astype(str)
        logging.info(f"Encoding non-numeric column: {column}")
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    features = data.drop(columns=["Label"], errors="ignore")
    labels = data["Label"]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    return features, labels, label_encoders, scaler

def add_noise_and_outliers(features, labels, noise_level=0.01, outlier_fraction=0.01):
    logging.info("Adding Gaussian noise to the dataset...")
    noisy_features = features + noise_level * np.random.normal(size=features.shape)
    logging.info("Adding outliers to the dataset...")
    n_outliers = int(outlier_fraction * len(features))
    outlier_indices = np.random.choice(len(features), n_outliers, replace=False)
    low = np.min(features, axis=0)
    high = np.max(features, axis=0)
    range_vals = high - low
    epsilon = 1e-6
    range_vals[range_vals == 0] = epsilon
    outlier_features = np.random.uniform(low=low, high=low + range_vals,
                                         size=(n_outliers, features.shape[1]))
    noisy_features[outlier_indices] = outlier_features
    return noisy_features, labels

def visualize_clusters(features, labels, title="Clustering Results"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    # Simple color mapping for binary classification
    colors = {"Normal": "green", "Anomalous": "red"}
    
    # Create figure with high resolution
    plt.figure(figsize=(8, 6), dpi=300)
    
    for lab in np.unique(labels):
        idx = np.where(labels == lab)
        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    c=colors.get(lab, "blue"), label=lab, s=10)
    
    plt.title(title)
    # Bold only the x and y axis labels
    plt.xlabel("PCA Component 1", fontweight="bold")
    plt.ylabel("PCA Component 2", fontweight="bold")
    plt.legend()
    plt.grid(True)
    plt.savefig(title.replace(" ", "_") + ".png", dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", display_labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    # Plot with a high dpi figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    disp.plot(cmap="viridis", ax=ax)
    
    # Bold only the x and y axis labels on the confusion matrix
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
    
    ax.set_title(title)
    plt.savefig(title.replace(" ", "_") + ".png", dpi=300)
    plt.show()

def visualize_pairwise(data, labels, title):
    df = pd.DataFrame(data)
    df["Class"] = labels

    # Create the pairplot and capture the grid object
    g = sns.pairplot(df, diag_kind="kde", hue="Class", palette="viridis")

    # Bold the x and y labels for each axis in the grid
    for ax in g.axes.flatten():
        if ax is not None:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            ax.set_xlabel(xlabel, fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")

    # Make the legend title and labels bold (if the legend exists)
    if g._legend is not None:
        # Set the legend title to "Class" (or use the existing one)
        g._legend.set_title("Class", prop={'weight': 'bold'})
        
        # Make the legend text labels bold
        for text in g._legend.texts:
            text.set_fontweight('bold')

    plt.savefig(title.replace(" ", "_") + ".png", dpi=300)
    plt.show()


def measure_power_consumption():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W

#####################
# First Layer (Binary Classification)
#####################

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

def evaluate_first_layer_cv(features, binary_labels, cv_folds=5):
    clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, features, binary_labels, cv=skf)
    logging.info(f"First-layer CV scores: {scores}")
    logging.info(f"Mean CV accuracy (First Layer): {np.mean(scores)*100:.2f}%")
    return clf

def train_first_layer(features, binary_labels):
    evaluate_first_layer_cv(features, binary_labels, cv_folds=5)
    X_train, X_test, y_train, y_test = train_test_split(features, binary_labels, test_size=0.3,
                                                        stratify=binary_labels, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info("First Layer (Binary) Evaluation:")
    logging.info(classification_report(y_test, preds))
    plot_confusion_matrix(y_test, preds, title="First Layer: Confusion Matrix",
                          display_labels=["Normal", "Anomalous"])
    return clf, acc

#####################
# Second Layer (Multi-Class) with Unknown
#####################

def evaluate_second_layer_cv(features, labels, cv_folds=5):
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=skf)
    logging.info(f"Second-layer CV scores: {scores}")
    logging.info(f"Mean CV accuracy (Second Layer): {np.mean(scores)*100:.2f}%")
    return clf

def train_second_layer_with_unknown(features, multi_labels, threshold_unknown=0.7):
    """
    Train a multi-class classifier on anomalies.
    If no class probability exceeds threshold_unknown, label the sample as 'unknown'.
    """
    # Evaluate with CV
    evaluate_second_layer_cv(features, multi_labels, cv_folds=5)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, multi_labels,
                                                        test_size=0.3, stratify=multi_labels, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Probability-based unknown detection
    probas = clf.predict_proba(X_test)
    known_classes = clf.classes_
    y_pred = []
    for row in probas:
        if np.max(row) < threshold_unknown:
            y_pred.append("unknown")
        else:
            y_pred.append(known_classes[np.argmax(row)])
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    logging.info("Second Layer (Multi-Class with Unknown) Evaluation:")
    logging.info(classification_report(y_test, y_pred, zero_division=0))
    
    # Deduplicate 'unknown' in confusion matrix labels
    all_labels = list(known_classes)
    if "unknown" not in all_labels:
        all_labels.append("unknown")
    
    plot_confusion_matrix(y_test, y_pred,
                          title="Second Layer: Confusion Matrix (with Unknown)",
                          display_labels=all_labels)
    return clf, acc

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Create a high-resolution figure
    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(range(len(importances)), importances[indices], align="center", color="green")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title("Feature Importances")
    plt.xlabel("Feature", fontweight="bold")
    plt.ylabel("Importance", fontweight="bold")
    plt.grid(True)
    plt.savefig("Feature_Importances.png", dpi=300)
    plt.show()

#####################
# Main
#####################

def main():
    # 1. Load known data
    known_data = load_and_combine_datasets(DATASETS, encoding="latin-1")
    
    # 2. (Optional) Load unknown data
    unknown_data = load_unknown_dataset(UNKNOWN_DATA_PATH, encoding="latin-1")
    
    # 3. Combine
    full_data = pd.concat([known_data, unknown_data], ignore_index=True)
    logging.info(f"Full dataset (known + unknown) shape: {full_data.shape}")
    
    # 4. Preprocess
    features, labels, label_encoders, scaler = preprocess_data(full_data)
    
    # 5. Noise & outliers
    noisy_features, noisy_labels = add_noise_and_outliers(features, labels,
                                                          noise_level=0.01,
                                                          outlier_fraction=0.01)
    
    # 6. Binary labels for first layer
    label_encoder_for_label = label_encoders.get("Label")
    if label_encoder_for_label is None:
        logging.error("No LabelEncoder for 'Label' found.")
        return
    original_labels = label_encoder_for_label.inverse_transform(noisy_labels)
    binary_labels = np.array(["Normal" if lab == "normal_state" else "Anomalous" for lab in original_labels])
    
    # 7. Train first layer
    clf1, acc1 = train_first_layer(noisy_features, binary_labels)
    logging.info(f"First Layer Accuracy: {acc1*100:.2f}%")
    visualize_clusters(noisy_features, binary_labels, title="First Layer: Supervised Binary Classification")
    visualize_pairwise(noisy_features, binary_labels, title="First Layer: Pairwise Visualization")
    
    # 8. Second layer: only anomalies
    anomaly_indices = np.where(binary_labels == "Anomalous")[0]
    if len(anomaly_indices) == 0:
        logging.error("No anomalies found in the first layer!")
        return
    second_layer_features = noisy_features[anomaly_indices]
    second_layer_labels = original_labels[anomaly_indices]
    
    # Filter out any leftover "normal_state"
    valid_idx = [i for i, lab in enumerate(second_layer_labels) if lab != "normal_state"]
    if len(valid_idx) == 0:
        logging.error("No attack or unknown samples remain for second layer!")
        return
    second_layer_features = second_layer_features[valid_idx]
    second_layer_labels = second_layer_labels[valid_idx]
    distribution = pd.Series(second_layer_labels).value_counts().to_dict()
    logging.info(f"Second-layer label distribution: {distribution}")
    
    # 9. Train second layer with unknown detection
    clf2, acc2 = train_second_layer_with_unknown(second_layer_features, second_layer_labels, threshold_unknown=0.7)
    logging.info(f"Second Layer Accuracy: {acc2*100:.2f}%")
    
    # 10. Visualize feature importance & pairwise
    feature_names = full_data.columns.drop("Label")
    plot_feature_importance(clf2, feature_names)
    visualize_pairwise(second_layer_features, second_layer_labels, title="Second Layer: Pairwise Visualization")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error occurred: {e}")
