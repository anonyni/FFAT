
##############################################################################################################
#  Working Code
##############################################################################################################


# import os
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.cluster import DBSCAN
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, silhouette_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# import matplotlib.pyplot as plt
# import logging
# from joblib import dump, load
# from sklearn.decomposition import PCA


# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# # Load datasets
# DATASETS = {
#     "Normal": "datasets_2_improved/normal_state.csv",
#     "Overvoltage": "datasets_2_improved/overvoltage.csv",
#     "Buffer Overflow": "datasets_2_improved/buffer_overflow.csv",
#     "Denial of Service": "datasets_2_improved/denial_of_service_(dos).csv",
#     "Cross Site Scripting": "datasets_2_improved/cross-site_scripting_(xss).csv",
#     "Code Injection": "datasets_2_improved/code_injection.csv"

#     # "Normal": "datasets_3_improved/normal_state.csv",
#     # "Overvoltage": "datasets_3_improved/overvoltage.csv",
#     # "Buffer Overflow": "datasets_3_improved/buffer_overflow.csv",
#     # "Denial of Service": "datasets_3_improved/denial_of_service_(dos).csv",
#     # "Cross Site Scripting": "datasets_3_improved/cross-site_scripting_(xss).csv",
#     # "Code Injection": "datasets_3_improved/code_injection.csv"


# }

# def load_and_combine_datasets(dataset_paths):
#     dfs = []
#     for label, path in dataset_paths.items():
#         df = pd.read_csv(path)
#         df["Label"] = label  # Assign the actual class name
#         dfs.append(df)
#     return pd.concat(dfs, ignore_index=True)

# def preprocess_data(data):
#     data = data.drop(columns=["Timestamp"], errors="ignore")
#     label_encoders = {}
#     for column in data.select_dtypes(include=["object"]):
#         logging.info(f"Encoding non-numeric column: {column}")
#         le = LabelEncoder()
#         data[column] = le.fit_transform(data[column])
#         label_encoders[column] = le
#     features = data.drop(columns=["Label"], errors="ignore")
#     labels = data["Label"]
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)
#     return features, labels, label_encoders

# def add_noise_and_outliers(features, labels, noise_level=0.05, outlier_fraction=0.02):
#     logging.info("Adding Gaussian noise to the dataset...")
#     noisy_features = features + noise_level * np.random.normal(size=features.shape)

#     logging.info("Adding outliers to the dataset...")
#     n_outliers = int(outlier_fraction * len(features))
#     outlier_indices = np.random.choice(len(features), n_outliers, replace=False)
#     outlier_features = np.random.uniform(
#         low=np.min(features, axis=0),
#         high=np.max(features, axis=0),
#         size=(n_outliers, features.shape[1])
#     )
#     noisy_features[outlier_indices] = outlier_features

#     return noisy_features, labels

# def visualize_clusters(features, labels, title):
#     pca = PCA(n_components=2)
#     reduced_features = pca.fit_transform(features)
    
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="viridis", s=10)
#     plt.colorbar(scatter, label="Cluster")
#     plt.title(title)
#     plt.xlabel("Reduced Dimension 1 (PCA)")
#     plt.ylabel("Reduced Dimension 2 (PCA)")
#     plt.grid(True)
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap="viridis")
#     plt.title(title)
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def plot_feature_importance(model, feature_names):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]
    
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(importances)), importances[indices], align="center", color="green")
#     plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
#     plt.title("Feature Importances")
#     plt.xlabel("Feature")
#     plt.ylabel("Importance")
#     plt.grid(True)
#     plt.savefig("Feature_Importances.png")
#     plt.show()

# def visualize_pairwise(data, labels, title):
#     df = pd.DataFrame(data)
#     df["Class"] = labels  # Assign class names
#     sns.pairplot(df, diag_kind="kde", hue="Class", palette="viridis")
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def train_second_layer(features, labels, feature_names):
#     features_train, features_test, labels_train, labels_test = train_test_split(
#         features, labels, test_size=0.3, stratify=labels, random_state=42
#     )
    
#     rf_model = RandomForestClassifier(random_state=42)
#     rf_model.fit(features_train, labels_train)
#     predictions = rf_model.predict(features_test)
    
#     cm = confusion_matrix(labels_test, predictions)
#     accuracy = cm.diagonal().sum() / cm.sum()
    
#     plot_confusion_matrix(labels_test, predictions, title="Second Layer: Confusion Matrix")
    
#     report = classification_report(labels_test, predictions, target_names=DATASETS.keys())
#     logging.info("\nSecond Layer Evaluation:\n" + report)
#     logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
#     plot_feature_importance(rf_model, feature_names)
    
#     dump(rf_model, "second_layer_model.joblib")
#     return rf_model


# def main():
#     data = load_and_combine_datasets(DATASETS)
#     features, labels, label_encoders = preprocess_data(data)
    
#     noisy_features, noisy_labels = add_noise_and_outliers(features, labels)

#     # First Layer: Binary Classification (Normal vs Anomalous)
#     dbscan = DBSCAN(eps=1.0, min_samples=15)
#     cluster_labels = dbscan.fit_predict(noisy_features)

#     # Assign labels: "Normal" or "Anomalous"
#     class_mapping = {0: "Normal", -1: "Anomalous"}
#     binary_labels = np.array([class_mapping.get(label, "Anomalous") for label in cluster_labels])
    
#     visualize_clusters(noisy_features, [0 if lbl == "Normal" else 1 for lbl in binary_labels], 
#                        title="First Layer: Clustering Results (Normal vs Anomalous)")
    
#     visualize_pairwise(noisy_features, binary_labels, title="First Layer: Pairwise Visualization")

#     # Second Layer: Supervised Classification
#     anomaly_indices = binary_labels == "Anomalous"
#     second_layer_features = noisy_features[anomaly_indices]
#     second_layer_labels = np.array(noisy_labels)[anomaly_indices]

#     if len(set(second_layer_labels)) > 1:
#         feature_names = data.columns[:-1]
        
#         # Convert class index back to class names
#         class_label_mapping = {index: name for index, name in enumerate(DATASETS.keys())}
#         named_labels = np.array([class_label_mapping[label] for label in second_layer_labels])
        
#         rf_model = train_second_layer(second_layer_features, named_labels, feature_names)

#         visualize_pairwise(second_layer_features, named_labels, title="Second Layer: Pairwise Visualization")
#     else:
#         logging.warning("Not enough anomalies detected for second-layer classification.")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         logging.error(f"Error occurred: {e}")


#############################################################################################################
# Best model with enough factors
############################################################################################################

# import os
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.cluster import DBSCAN
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# from sklearn.model_selection import train_test_split, cross_val_score
# import matplotlib.pyplot as plt
# import logging
# from joblib import dump, load
# from sklearn.decomposition import PCA
# import time
# import tracemalloc
# import pynvml

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# # Load datasets
# DATASETS = {
#     "normal_state": "datasets_2_improved/normal_state.csv",
#     "overvoltage": "datasets_2_improved/overvoltage.csv",
#     "buffer_overflow": "datasets_2_improved/buffer_overflow.csv",
#     "denial_of_service": "datasets_2_improved/denial_of_service_(dos).csv",
#     "cross_site_scripting": "datasets_2_improved/cross-site_scripting_(xss).csv",
#     "code_injection": "datasets_2_improved/code_injection.csv"
# }

# def load_and_combine_datasets(dataset_paths):
#     dfs = []
#     for label, path in dataset_paths.items():
#         df = pd.read_csv(path)
#         df["Label"] = label
#         dfs.append(df)
#     return pd.concat(dfs, ignore_index=True)

# def preprocess_data(data):
#     data = data.drop(columns=["Timestamp"], errors="ignore")
#     label_encoders = {}
#     for column in data.select_dtypes(include=["object"]):
#         logging.info(f"Encoding non-numeric column: {column}")
#         le = LabelEncoder()
#         data[column] = le.fit_transform(data[column])
#         label_encoders[column] = le
#     features = data.drop(columns=["Label"], errors="ignore")
#     labels = data["Label"]
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)
#     return features, labels, label_encoders

# def add_noise_and_outliers(features, labels, noise_level=0.05, outlier_fraction=0.02):
#     logging.info("Adding Gaussian noise to the dataset...")
#     noisy_features = features + noise_level * np.random.normal(size=features.shape)

#     logging.info("Adding outliers to the dataset...")
#     n_outliers = int(outlier_fraction * len(features))
#     outlier_indices = np.random.choice(len(features), n_outliers, replace=False)
#     outlier_features = np.random.uniform(
#         low=np.min(features, axis=0),
#         high=np.max(features, axis=0),
#         size=(n_outliers, features.shape[1])
#     )
#     noisy_features[outlier_indices] = outlier_features

#     return noisy_features, labels

# def visualize_clusters(features, labels, title="Clustering Results"):
#     # PCA for dimensionality reduction
#     pca = PCA(n_components=2)
#     reduced_features = pca.fit_transform(features)
    
#     # Map 'Normal' and 'Anomalous' to numeric values for plotting
#     numeric_labels = np.where(labels == "Normal", 0, 1)
    
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=numeric_labels, cmap="viridis", s=10)
#     plt.colorbar(scatter, label="Cluster")
#     plt.title(title)
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.grid(True)
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()


# def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap="viridis")
#     plt.title(title)
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def plot_feature_importance(model, feature_names):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(importances)), importances[indices], align="center", color="green")
#     plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
#     plt.title("Feature Importances")
#     plt.xlabel("Feature")
#     plt.ylabel("Importance")
#     plt.grid(True)
#     plt.savefig("Feature_Importances.png")
#     plt.show()

# def visualize_pairwise(data, labels, title="Pairwise Plot"):
#     data = pd.DataFrame(data)
#     data["Class"] = labels
#     sns.pairplot(data, diag_kind="kde", hue="Class", palette="viridis")
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def measure_scalability(model, datasets):
#     results = []
#     for size, (X, y) in datasets.items():
#         tracemalloc.start()
#         start_time = time.time()
#         model.fit(X, y)
#         train_time = time.time() - start_time
#         current, peak = tracemalloc.get_traced_memory()
#         tracemalloc.stop()
#         results.append({
#             "Dataset Size": size,
#             "Training Time (s)": train_time,
#             "Peak Memory Usage (MB)": peak / (1024 * 1024)
#         })
#     return pd.DataFrame(results)

# def measure_latency(model, X_test):
#     batch_sizes = [1, 10, 100, 1000]
#     latency_results = []
#     for batch_size in batch_sizes:
#         batch = X_test[:batch_size]
#         start_time = time.time()
#         model.predict(batch)
#         latency = time.time() - start_time
#         latency_results.append({"Batch Size": batch_size, "Latency (s)": latency / batch_size})
#     return pd.DataFrame(latency_results)

# def train_second_layer(features, labels):
#     features_train, features_test, labels_train, labels_test = train_test_split(
#         features, labels, test_size=0.3, stratify=labels, random_state=42
#     )
#     rf_model = RandomForestClassifier(random_state=42)
#     rf_model.fit(features_train, labels_train)
#     predictions = rf_model.predict(features_test)
#     cm = confusion_matrix(labels_test, predictions)
#     accuracy = cm.diagonal().sum() / cm.sum()
#     plot_confusion_matrix(labels_test, predictions, title="Second Layer: Confusion Matrix")
#     report = classification_report(labels_test, predictions)
#     logging.info("\nSecond Layer Evaluation:\n" + report)
#     logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")
#     dump(rf_model, "second_layer_model.joblib")
#     return rf_model

# def main():
#     data = load_and_combine_datasets(DATASETS)
#     features, labels, label_encoders = preprocess_data(data)
#     noisy_features, noisy_labels = add_noise_and_outliers(features, labels)
#     dbscan = DBSCAN(eps=1.0, min_samples=15)
#     cluster_labels = dbscan.fit_predict(noisy_features)
#     binary_labels = np.where(cluster_labels == 0, "Normal", "Anomalous")
#     visualize_clusters(noisy_features, binary_labels, title="First Layer: Clustering Results")
#     visualize_pairwise(noisy_features, binary_labels, title="First Layer: Pairwise Plot")
#     anomaly_indices = binary_labels == "Anomalous"
#     second_layer_features = noisy_features[anomaly_indices]
#     second_layer_labels = noisy_labels[anomaly_indices]
#     if len(set(second_layer_labels)) > 1:
#         rf_model = train_second_layer(second_layer_features, second_layer_labels)
#         feature_names = data.columns[:-1]
#         plot_feature_importance(rf_model, feature_names)
#         scalability_results = measure_scalability(rf_model, {"10k": (features[:10000], labels[:10000])})
#         latency_results = measure_latency(rf_model, second_layer_features)
#         logging.info("\nScalability Results:\n" + str(scalability_results))
#         logging.info("\nLatency Results:\n" + str(latency_results))
#     else:
#         logging.warning("Not enough anomalies detected for second-layer classification.")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         logging.error(f"Error occurred: {e}")

#################################################################################################################################
# New one #
##################################################################################################################################
# import os
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.cluster import DBSCAN
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import logging
# from joblib import dump, load
# from sklearn.decomposition import PCA
# import time
# import tracemalloc
# import pynvml

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# # Load datasets
# DATASETS = {
#     "normal_state": "datasets_2_improved/normal_state.csv",
#     "overvoltage": "datasets_2_improved/overvoltage.csv",
#     "buffer_overflow": "datasets_2_improved/buffer_overflow.csv",
#     "denial_of_service": "datasets_2_improved/denial_of_service_(dos).csv",
#     "cross_site_scripting": "datasets_2_improved/cross-site_scripting_(xss).csv",
#     "code_injection": "datasets_2_improved/code_injection.csv"
# }

# # Define anomaly type mapping for visualization
# CLASS_MAPPING = {
#     0: "Normal",
#     1: "Overvoltage",
#     2: "Buffer Overflow",
#     3: "Denial of Service",
#     4: "Cross-Site Scripting",
#     5: "Code Injection"
# }

# def load_and_combine_datasets(dataset_paths):
#     dfs = []
#     for label, path in dataset_paths.items():
#         df = pd.read_csv(path)
#         df["Label"] = label
#         dfs.append(df)
#     return pd.concat(dfs, ignore_index=True)

# def preprocess_data(data):
#     data = data.drop(columns=["Timestamp"], errors="ignore")
#     label_encoders = {}
#     for column in data.select_dtypes(include=["object"]):
#         logging.info(f"Encoding non-numeric column: {column}")
#         le = LabelEncoder()
#         data[column] = le.fit_transform(data[column])
#         label_encoders[column] = le
#     features = data.drop(columns=["Label"], errors="ignore")
#     labels = data["Label"]
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)
#     return features, labels, label_encoders

# def add_noise_and_outliers(features, labels, noise_level=0.05, outlier_fraction=0.02):
#     logging.info("Adding Gaussian noise to the dataset...")
#     noisy_features = features + noise_level * np.random.normal(size=features.shape)

#     logging.info("Adding outliers to the dataset...")
#     n_outliers = int(outlier_fraction * len(features))
#     outlier_indices = np.random.choice(len(features), n_outliers, replace=False)
#     outlier_features = np.random.uniform(
#         low=np.min(features, axis=0),
#         high=np.max(features, axis=0),
#         size=(n_outliers, features.shape[1])
#     )
#     noisy_features[outlier_indices] = outlier_features

#     return noisy_features, labels

# def visualize_clusters(features, labels, title="Clustering Results"):
#     pca = PCA(n_components=2)
#     reduced_features = pca.fit_transform(features)

#     # Convert categorical labels to numerical for visualization
#     unique_labels = list(set(labels))
#     label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
#     numeric_labels = np.array([label_to_numeric[label] for label in labels])

#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=numeric_labels, cmap="viridis", s=10)
#     plt.colorbar(scatter, ticks=range(len(unique_labels)), label="Cluster")
#     plt.title(title)
#     plt.xlabel("Reduced Dimension 1")
#     plt.ylabel("Reduced Dimension 2")
#     plt.grid(True)
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap="viridis")
#     plt.title(title)
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def plot_feature_importance(model, feature_names):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(importances)), importances[indices], align="center", color="green")
#     plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
#     plt.title("Feature Importances")
#     plt.xlabel("Feature")
#     plt.ylabel("Importance")
#     plt.grid(True)
#     plt.savefig("Feature_Importances.png")
#     plt.show()


# def visualize_pairwise(data, labels, title):
#     df = pd.DataFrame(data)
#     df["Class"] = labels  # Assign class names
#     sns.pairplot(df, diag_kind="kde", hue="Class", palette="viridis")
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def measure_power_consumption():
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert mW to W
#     return power_draw

# def train_second_layer(features, labels):
#     features_train, features_test, labels_train, labels_test = train_test_split(
#         features, labels, test_size=0.3, stratify=labels, random_state=42
#     )
#     rf_model = RandomForestClassifier(random_state=42)

#     start_time = time.time()
#     rf_model.fit(features_train, labels_train)
#     training_time = time.time() - start_time
#     power = measure_power_consumption()

#     predictions = rf_model.predict(features_test)
#     accuracy = accuracy_score(labels_test, predictions)
#     energy_consumption = power * training_time  # Energy (Joules)

#     plot_confusion_matrix(labels_test, predictions, title="Second Layer: Confusion Matrix")
#     report = classification_report(labels_test, predictions)
#     logging.info("\nSecond Layer Evaluation:\n" + report)
#     logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")
#     logging.info(f"Training Time: {training_time:.4f} s")
#     logging.info(f"Power Consumption: {power:.2f} W")
#     logging.info(f"Total Energy Used: {energy_consumption:.4f} J")

#     dump(rf_model, "second_layer_model.joblib")
#     return rf_model

# def main():
#     data = load_and_combine_datasets(DATASETS)
#     features, labels, label_encoders = preprocess_data(data)

#     noisy_features, noisy_labels = add_noise_and_outliers(features, labels)

#     dbscan = DBSCAN(eps=1.0, min_samples=15)
#     cluster_labels = dbscan.fit_predict(noisy_features)

#     binary_labels = np.where(cluster_labels == 0, "Normal", "Anomalous")
#     visualize_clusters(noisy_features, binary_labels, title="First Layer: Clustering Results")
#     visualize_pairwise(noisy_features, binary_labels, title="First Layer: Pairwise Visualization")

#     anomaly_indices = binary_labels == "Anomalous"
#     second_layer_features = noisy_features[anomaly_indices]
#     second_layer_labels = noisy_labels[anomaly_indices]

#     if len(set(second_layer_labels)) > 1:
#         # Convert class index back to class names
#         class_label_mapping = {index: name for index, name in enumerate(DATASETS.keys())}
#         named_labels = np.array([class_label_mapping[label] for label in second_layer_labels])
#         rf_model = train_second_layer(second_layer_features, named_labels)
#         feature_names = data.columns[:-1]
#         plot_feature_importance(rf_model, feature_names)
#         #scalability_results = measure_scalability(rf_m odel, {"10k": (features[:10000], labels[:10000])})
#         visualize_pairwise(second_layer_features, named_labels, title="Second Layer: Pairwise Visualization")
#     else:
#         logging.warning("Not enough anomalies detected for second-layer classification.")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         logging.error(f"Error occurred: {e}")

######################################################################################################################################

                                          ### Main and Final Code ###

######################################################################################################################################

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
import tracemalloc
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# DATASETS dictionary (adjust file paths as needed)
DATASETS = {
    "normal_state": "attack_data/normal_state.csv",
    "overvoltage": "attack_data/overvoltage.csv",
    "clock_glitching": "attack_data/clock_glitching.csv",
    "rowhammer": "attack_data/rowhammer.csv"
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

def add_noise_and_outliers(features, labels, noise_level=0.06, outlier_fraction=0.04):
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
    plt.savefig(title.replace(" ", "_") + ".png", dpi=200)
    # plt.savefig(title.replace(" ", "_") + ".png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", display_labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    # Plot with a high dpi figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    # fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="viridis", ax=ax)
    
    # Bold only the x and y axis labels on the confusion matrix
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
    
    ax.set_title(title)
    plt.savefig(title.replace(" ", "_") + ".png", dpi=200)
    # plt.savefig(title.replace(" ", "_") + ".png")
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

    # plt.savefig(title.replace(" ", "_") + ".png", dpi=200)
    # # plt.savefig(title.replace(" ", "_") + ".png")
    # plt.show()


# def measure_power_consumption():
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W


def measure_scalability(model, X, y, batch_sizes=[100, 1000, 10000]):
    results = []
    for batch_size in batch_sizes:
        X_batch, y_batch = X[:batch_size], y[:batch_size]

        tracemalloc.start()
        cpu_power_start, gpu_power_start = measure_power_consumption()
        start_time = time.time()
        model.fit(X_batch, y_batch)
        training_time = time.time() - start_time
        cpu_power_end, gpu_power_end = measure_power_consumption()
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            "Batch Size": batch_size,
            "Training Time (s)": training_time,
            "Peak Memory (MB)": peak_memory / (1024 * 1024),
            "CPU Power (W)":  abs(cpu_power_end - cpu_power_start),
            "GPU Power (W)":  abs(gpu_power_start - gpu_power_end)
            # "Energy Consumption (J)" : energy_consumption
        })
    return pd.DataFrame(results)

def measure_power_consumption():
    """Measure CPU and GPU power consumption in watts"""
    time.sleep(0.1)  # Give time for CPU load to update
    cpu_power = psutil.cpu_percent(interval=0.1) * psutil.cpu_count() / 100
    gpu_power = 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
        pynvml.nvmlShutdown()
    except Exception:
        logging.warning("GPU power measurement failed. Running on CPU.")
    return cpu_power, gpu_power

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


    # Track resources
    tracemalloc.start()
    cpu_power_start, gpu_power_start = measure_power_consumption()
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    cpu_power_end, gpu_power_end = measure_power_consumption()
    current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
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
    # importances = model.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # # Create a high-resolution figure
    # # plt.figure(figsize=(8, 6), dpi=300)
    # plt.figure(figsize=(8, 6))
    # plt.bar(range(len(importances)), importances[indices], align="center", color="green")
    # plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    # plt.title("Feature Importances")
    # plt.xlabel("Feature", fontweight="bold")
    # plt.ylabel("Importance", fontweight="bold")
    # plt.grid(True)
    # plt.savefig("Feature_Importances.png")
    # plt.show()

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
    plt.figure(figsize=(8, 6), dpi=200)
    plt.bar(range(len(sorted_importances)), sorted_importances, color="green")
    plt.xticks(range(len(sorted_importances)), friendly_names, rotation=60)
    plt.xlabel("Feature", fontweight="bold")
    plt.ylabel("Importance", fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Feature_Importances_Model_Sorted.png", dpi=200)
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
                                                          noise_level=0.06,
                                                          outlier_fraction=0.04)
    
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
    scalability_results = measure_scalability(clf2, second_layer_features, second_layer_labels)
    logging.info("\nScalability Results:\n" + str(scalability_results))
    
    # 10. Visualize feature importance & pairwise
    feature_names = full_data.columns.drop("Label")
    plot_feature_importance(clf2, feature_names)
    visualize_pairwise(second_layer_features, second_layer_labels, title="Second Layer: Pairwise Visualization")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error occurred: {e}")




############################################################################################################################
# import os
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.cluster import DBSCAN
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import logging
# from joblib import dump
# from sklearn.decomposition import PCA
# import time
# import tracemalloc
# import pynvml
# import psutil

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# # Load datasets
# DATASETS = {
#     "normal_state": "datasets_2_improved/normal_state.csv",
#     "overvoltage": "datasets_2_improved/overvoltage.csv",
#     "buffer_overflow": "datasets_2_improved/buffer_overflow.csv",
#     "denial_of_service": "datasets_2_improved/denial_of_service_(dos).csv",
#     "cross_site_scripting": "datasets_2_improved/cross-site_scripting_(xss).csv",
#     "code_injection": "datasets_2_improved/code_injection.csv"
# }

# # Define anomaly type mapping for visualization
# CLASS_MAPPING = {
#     0: "Normal",
#     1: "Overvoltage",
#     2: "Buffer Overflow",
#     3: "Denial of Service",
#     4: "Cross-Site Scripting",
#     5: "Code Injection"
# }

# def load_and_combine_datasets(dataset_paths):
#     dfs = []
#     for label, path in dataset_paths.items():
#         df = pd.read_csv(path)
#         df["Label"] = label
#         dfs.append(df)
#     return pd.concat(dfs, ignore_index=True)

# def preprocess_data(data):
#     data = data.drop(columns=["Timestamp"], errors="ignore")
#     label_encoders = {}
#     for column in data.select_dtypes(include=["object"]):
#         logging.info(f"Encoding non-numeric column: {column}")
#         le = LabelEncoder()
#         data[column] = le.fit_transform(data[column])
#         label_encoders[column] = le
#     features = data.drop(columns=["Label"], errors="ignore")
#     labels = data["Label"]
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)
#     return features, labels, label_encoders


# def add_noise_and_outliers(features, labels, noise_level=0.05, outlier_fraction=0.02):
#     logging.info("Adding Gaussian noise to the dataset...")
#     noisy_features = features + noise_level * np.random.normal(size=features.shape)

#     logging.info("Adding outliers to the dataset...")
#     n_outliers = int(outlier_fraction * len(features))
#     outlier_indices = np.random.choice(len(features), n_outliers, replace=False)
#     outlier_features = np.random.uniform(
#         low=np.min(features, axis=0),
#         high=np.max(features, axis=0),
#         size=(n_outliers, features.shape[1])
#     )
#     noisy_features[outlier_indices] = outlier_features

#     return noisy_features, labels


# def visualize_clusters(features, labels, title="Clustering Results"):
#     pca = PCA(n_components=2)
#     reduced_features = pca.fit_transform(features)

#     # Convert categorical labels to numerical for visualization
#     unique_labels = list(set(labels))
#     label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
#     numeric_labels = np.array([label_to_numeric[label] for label in labels])

#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=numeric_labels, cmap="viridis", s=10)
#     plt.colorbar(scatter, ticks=range(len(unique_labels)), label="Cluster")
#     plt.title(title)
#     plt.xlabel("Reduced Dimension 1")
#     plt.ylabel("Reduced Dimension 2")
#     plt.grid(True)
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap="viridis")
#     plt.title(title)
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def plot_feature_importance(model, feature_names):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(importances)), importances[indices], align="center", color="green")
#     plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
#     plt.title("Feature Importances")
#     plt.xlabel("Feature")
#     plt.ylabel("Importance")
#     plt.grid(True)
#     plt.savefig("Feature_Importances.png")
#     plt.show()


# def visualize_pairwise(data, labels, title):
#     df = pd.DataFrame(data)
#     df["Class"] = labels  # Assign class names
#     sns.pairplot(df, diag_kind="kde", hue="Class", palette="viridis")
#     plt.savefig(title.replace(" ", "_") + ".png")
#     plt.show()

# def measure_power():
#     """Measure CPU and GPU power consumption in watts"""
#     time.sleep(0.1)  # Give time for CPU load to update
#     cpu_power = psutil.cpu_percent(interval=0.1) * psutil.cpu_count() / 100
#     gpu_power = 0
#     try:
#         pynvml.nvmlInit()
#         handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#         gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
#         pynvml.nvmlShutdown()
#     except Exception:
#         logging.warning("GPU power measurement failed. Running on CPU.")
#     return cpu_power, gpu_power


# def measure_scalability(model, X, y, batch_sizes=[100, 1000, 10000]):
#     results = []
#     for batch_size in batch_sizes:
#         X_batch, y_batch = X[:batch_size], y[:batch_size]

#         tracemalloc.start()
#         cpu_power_start, gpu_power_start = measure_power()
#         start_time = time.time()
#         model.fit(X_batch, y_batch)
#         training_time = time.time() - start_time
#         cpu_power_end, gpu_power_end = measure_power()
#         current, peak_memory = tracemalloc.get_traced_memory()
#         tracemalloc.stop()

#         results.append({
#             "Batch Size": batch_size,
#             "Training Time (s)": training_time,
#             "Peak Memory (MB)": peak_memory / (1024 * 1024),
#             "CPU Power (W)":  abs(cpu_power_end - cpu_power_start),
#             "GPU Power (W)":  abs(gpu_power_start - gpu_power_end)
#             # "Energy Consumption (J)" : energy_consumption
#         })
#     return pd.DataFrame(results)

# def train_second_layer(features, labels):
#     features_train, features_test, labels_train, labels_test = train_test_split(
#         features, labels, test_size=0.3, stratify=labels, random_state=42
#     )
#     rf_model = RandomForestClassifier(random_state=42)
    
#     # Track resources
#     tracemalloc.start()
#     cpu_power_start, gpu_power_start = measure_power()
#     start_time = time.time()
#     rf_model.fit(features_train, labels_train)
#     training_time = time.time() - start_time
#     cpu_power_end, gpu_power_end = measure_power()
#     current, peak_memory = tracemalloc.get_traced_memory()
#     tracemalloc.stop()

#     predictions = rf_model.predict(features_test)
#     accuracy = sum(predictions == labels_test) / len(labels_test)
#     energy_consumption = abs(cpu_power_start - cpu_power_end) * training_time

#     cm = confusion_matrix(labels_test, predictions)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap="viridis")
#     plt.title("Second Layer: Confusion Matrix")
#     plt.savefig("Confusion_Matrix.png")
#     plt.show()

#     CPU_power = abs(cpu_power_end - cpu_power_start)
#     GPU_power = abs(gpu_power_end - gpu_power_start)

#     report = classification_report(labels_test, predictions)
#     logging.info("\nSecond Layer Evaluation:\n" + report)
#     logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")
#     logging.info(f"Training Time: {training_time:.3f} sec, Peak Memory: {peak_memory / (1024 * 1024):.3f} MB")
#     logging.info(f"CPU Power Consumption: {CPU_power:.2f} W")
#     logging.info(f"GPU Power Consumption: {GPU_power:.2f} W")
#     logging.info(f"Total Energy Used: {energy_consumption:.4f} J")

#     dump(rf_model, "second_layer_model.joblib")
#     return rf_model

# def main():
#     data = load_and_combine_datasets(DATASETS)
#     features, labels, label_encoders = preprocess_data(data)
#     noisy_features, noisy_labels = add_noise_and_outliers(features, labels)

#     dbscan = DBSCAN(eps=1.0, min_samples=15)
#     cluster_labels = dbscan.fit_predict(features)

#     cluster_mapping = {0: "Normal", -1: "Anomalous"}
#     binary_labels = np.array([cluster_mapping.get(label, "Anomalous") for label in cluster_labels])
#     visualize_clusters(noisy_features, binary_labels, title="First Layer: Clustering Results")
#     visualize_pairwise(noisy_features, binary_labels, title="First Layer: Pairwise Visualization")

#     anomaly_indices = binary_labels == "Anomalous"
#     second_layer_features = noisy_features[anomaly_indices]
#     second_layer_labels = noisy_labels[anomaly_indices]


#     if len(set(second_layer_labels)) > 1:
#         # Convert class index back to class names
#         class_label_mapping = {index: name for index, name in enumerate(DATASETS.keys())}
#         named_labels = np.array([class_label_mapping[label] for label in second_layer_labels])

#         rf_model = train_second_layer(second_layer_features, named_labels)
#         feature_names = data.columns[:-1]
#         plot_feature_importance(rf_model, feature_names)
#         scalability_results = measure_scalability(rf_model, features, labels)
#         logging.info("\nScalability Results:\n" + str(scalability_results))
#         visualize_pairwise(second_layer_features, named_labels, title="Second Layer: Pairwise Visualization")
#     else:
#         logging.warning("Not enough anomalies detected for second-layer classification.")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         logging.error(f"Error occurred: {e}")

































