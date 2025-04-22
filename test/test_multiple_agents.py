import json
import logging
import os
from typing import List, Optional, Tuple, Union 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import tensorflow as tf
import time

from data.cic_data_manager import CICDataManager
from data.nsl_kdd_data_manager import NslKddDataManager
from utils.config import NSL_KDD_FORMATTED_TEST_PATH, NSL_KDD_FORMATTED_TRAIN_PATH, ORIGINAL_KDD_TEST, ORIGINAL_KDD_TRAIN
from utils.helpers import calculate_f1_scores_per_class_dynamically, calculate_general_overview_per_attack_type, calculate_one_vs_all_metrics, get_cf_matrix, get_model_summary
from utils.plotting import plot_confusion_matrix, plot_roc_curve
from utils.plotting_multiple_agents import visualize_q_value_errors

def test_trained_agent_quality_on_intra_set(path_to_model, data_mgr: Union[NslKddDataManager | CICDataManager], plots_path, **kwargs):
    one_vs_all = kwargs.get('one_vs_all', False)
    metrics_json = {}
    if one_vs_all:
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

    model = tf.keras.models.load_model(path_to_model)
    states, labels = (data_mgr.get_test_set() if isinstance(data_mgr, CICDataManager)
                      else data_mgr.get_full())
    
    start_time=time.time()
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    q = model.predict(states_tensor)
    actions = np.argmax(q, axis=1) # get the action with the highest Q-value -> the predicted attack type

    true_attack_type_indices = (get_cic_true_attack_type_indices(labels, data_mgr, one_vs_all)
                            if isinstance(data_mgr, CICDataManager)
                            else get_nsl_kdd_true_attack_type_indices(labels, data_mgr))

    plot_confusion_matrix(get_cf_matrix(true_attack_type_indices, actions), 
                        classes=data_mgr.attack_types, path=os.path.join(plots_path, 'intraset/'), normalize=True,
                        title='Normalized confusion matrix')
    
    logging.info(f"Model summary:\n{get_model_summary(model)}\nloaded from: {path_to_model}")
    
    
    total_reward = 0
    true_labels = np.zeros(len(data_mgr.attack_types),dtype=int)
    predicted_labels = np.zeros(len(data_mgr.attack_types),dtype=int)
    predicted_correct_labels = np.zeros(len(data_mgr.attack_types),dtype=int)
    labels_per_attack_type, counts = np.unique(true_attack_type_indices, return_counts=True)
    true_labels[labels_per_attack_type] += counts

    metrics = calculate_tp_tn_fp_fn(true_attack_type_indices, actions, len(data_mgr.attack_types))
    for indx, prediction in enumerate(actions): 
        predicted_labels[prediction] +=1
        true = true_attack_type_indices[indx]
        if prediction == true: # correct prediction
            total_reward += 1
            predicted_correct_labels[prediction] += 1
            if one_vs_all: # One vs All: 0=normal, 1=attack
                if true == 1: # attack predicted as attack
                    true_positive += 1
                else: # normal predicted normal
                    true_negative += 1
        else: # wrong prediction
            if one_vs_all: # One vs All: 0=normal, 1=attack
                if prediction == 1: # actual normal predicted as attack
                    false_positive += 1
                else: # actual attack predicted as normal
                    false_negative += 1

    f1_scores_dynamically = calculate_f1_scores_per_class_dynamically(actions, data_mgr.attack_types, true_attack_type_indices)
    mismatch = predicted_labels - true_labels
    acc = float(100 * total_reward / len(states))

    if one_vs_all:
        # One vs All: 0=normal, 1=attack
        accuracy, recall, specificity, precision, f1_score, fpr = calculate_metrics(true_positive, true_negative, false_positive, false_negative)
        logging.info(f"One vs All metric calculation: \nTrue positive: {true_positive} | False positive: {false_positive} | \nTrue negative: {true_negative} | False negative: {false_negative}" + 
                     f"\nAccuracy: {accuracy:.4f} | Recall: {recall:.4f} | Specificity: {specificity:.4f} | Precision: {precision:.4f} | F1-Score: {f1_score:.4f} | False Positive Rate: {fpr:.4f}")
        metrics_json["one_vs_all"] = {
            "tp": true_positive, "tn": true_negative, "fp": false_positive, "fn": false_negative,
            "accuracy": accuracy, "recall": recall, "specificity": specificity,
            "precision": precision, "f1_score": f1_score, "false_positive_rate": fpr
        }
    
    outputs_df = calculate_general_overview_per_attack_type(data_mgr.attack_types, predicted_labels, predicted_correct_labels, true_labels, f1_scores_dynamically, mismatch)
    logging.info(f"Overall overview\nTotal reward: {total_reward} | Number of samples: {len(states)} | Accuracy = {acc:.2f}%\nOverall overview per attack type\n{outputs_df}")
    metrics_df, weighted_f1 = calculate_f1_overview(true_attack_type_indices, actions, data_mgr.attack_types)
    logging.info("\n" + metrics_df.to_string(index=False))
    logging.info(f"Overall weighted F1-score: {weighted_f1:.4f}")
    #if not one_vs_all:
    perf_per_class = calculate_one_vs_all_metrics(true_attack_type_indices, actions, attack_type=data_mgr.attack_types)
    logging.info(f"\r\nOne vs All metrics: \r\n{perf_per_class}")
    metrics_json["one_vs_all_detailed"] = perf_per_class.to_dict(orient="records")
    report = classification_report(true_attack_type_indices, actions, target_names=data_mgr.attack_types)

    logging.info(f"Classification report:\n{report}")
    loss, mse, mae, _, precision, recall, _ = model.evaluate(states_tensor, pd.get_dummies(true_attack_type_indices), verbose=2)
    logging.info(f"Model metrics: \nloss={loss}, mse={mse}, mae={mae}")
    logging.info(f"Optimizer config: {model.optimizer.get_config()}")
    logging.info(f"Time needed for testing: {time.time() - start_time}")
    metrics_json['f1_per_class'] = metrics_df.to_dict(orient='records')
    metrics_json['weighted_f1'] = weighted_f1
    report_json = classification_report(true_attack_type_indices, actions, target_names=data_mgr.attack_types, output_dict=True)
    metrics_json["classification_report"] = report_json
    metrics_json["model_eval"] = {"loss": loss, "mse": mse, "mae": mae}
    metrics_json["optimizer"] = model.optimizer.get_config()
    metrics_json["attack_types"] = data_mgr.attack_types
    metrics_json['metrics'] = metrics
    json_path = os.path.join(os.path.dirname(path_to_model), "logs", f"evaluation_metrics_{model.name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=4, default=convert_numpy)
    logging.info(f"Metrics exported to: {json_path}")


def test_trained_agent_quality_on_inter_set(path_to_model: str, 
                                            x_test: np.ndarray, 
                                            y_test: np.ndarray,
                                            plots_path: str, 
                                            one_vs_all: bool = True,
                                            attack_types: Optional[List[str]] = None):

    start_time = time.time()
    metrics_json = {}
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

    model = tf.keras.models.load_model(path_to_model)
    states_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    q = model.predict(states_tensor)
    predictions = np.argmax(q, axis=1)

    label_mapping = {type: index for index, type in enumerate(attack_types)}
    true_labels = np.array(y_test.replace(label_mapping).astype(int).values.flatten())

    if one_vs_all:
        metrics = calculate_tp_tn_fp_fn(true_labels, predictions, len(attack_types))
        true_positive, true_negative, false_positive, false_negative = metrics[1]['TP'], metrics[1]['TN'], metrics[1]['FP'], metrics[1]['FN']

        accuracy, recall, specificity, precision, f1_score_val, fpr = calculate_metrics(
            true_positive, true_negative, false_positive, false_negative
        )

        metrics_json["one_vs_all"] = {
            "tp": true_positive, "tn": true_negative, "fp": false_positive, "fn": false_negative,
            "accuracy": accuracy, "recall": recall, "specificity": specificity,
            "precision": precision, "f1_score": f1_score_val, "false_positive_rate": fpr
        }

        logging.info(f"[Intra-Set] Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, "
                     f"Precision: {precision:.4f}, F1: {f1_score_val:.4f}")

    # Optional: Confusion Matrix
    if attack_types:
        cf = get_cf_matrix(true_labels, predictions)
        plot_confusion_matrix(cf,
                            classes=attack_types,
                            path=os.path.join(plots_path, 'interset/'),
                            normalize=True,
                            title="Normalized confusion matrix (Intra-Set)")
        logging.info(f"[Intra-Set] Confusion matrix saved to {plots_path}")

        report = classification_report(true_labels, predictions, target_names=attack_types, output_dict=True)
        metrics_json["classification_report"] = report

    # Modell-Metriken
    loss, mse, mae, _, precision, recall, _ = model.evaluate(states_tensor, pd.get_dummies(true_labels), verbose=2)
    logging.info(f"[Cross-Set] Model metrics: loss={loss}, mse={mse}, mae={mae}")

    metrics_json["model_eval"] = {"loss": loss, "mse": mse, "mae": mae}
    metrics_json["optimizer"] = model.optimizer.get_config()
    metrics_json["attack_types"] = attack_types or ["Benign", "Attack"]

    # Export JSON
    json_path = os.path.join(os.path.dirname(path_to_model), "logs", f"evaluation_metrics_cross_{model.name}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=4, default=convert_numpy)

    logging.info(f"[Intra-Set] Evaluation completed. Metrics exported to: {json_path}")
    logging.info(f"[Intra-Set] Time needed: {time.time() - start_time:.2f} seconds")


def calculate_f1_overview(true_labels, predicted_labels, class_names: list[str]) -> Tuple[pd.DataFrame, float]:
    """
    Berechnet Precision, Recall, F1 und Support für jede Klasse sowie den gewichteten F1-Score.
    
    Args:
        true_labels: Liste oder Array der Ground Truth.
        predicted_labels: Liste oder Array der Modellvorhersagen.
        class_names: Reihenfolge der Klassenbezeichner (z. B. ['normal', 'DoS', 'Probe', ...]).

    Returns:
        Tuple: (DataFrame mit Per-Klasse-Metriken, gewichteter F1-Score)
    """
    # Support = Anzahl pro Klasse in true_labels
    support = pd.Series(true_labels).value_counts().reindex(range(len(class_names)), fill_value=0).values

    f1s = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
    recalls = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
    precisions = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
    accuracies = [(true_labels == predicted_labels)[np.array(true_labels) == i].mean() for i in range(len(class_names))]

    per_class_df = pd.DataFrame({
        'Class': class_names,
        'Support': support,
        'Accuracy': np.round(accuracies, 4),
        'Precision': np.round(precisions, 4),
        'Recall': np.round(recalls, 4),
        'F1-Score': np.round(f1s, 4)
    })

    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    return per_class_df, weighted_f1

def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def get_nsl_kdd_true_attack_type_indices(labels: pd.DataFrame, data_mgr: NslKddDataManager) -> list:
    true_attack_type_indices=[] # list of true attack types as indices (0-4). 0=normal, 1=dos, 2=probe, 3=r2l, 4=u2r. Length = number of samples
    for _, label in labels.iterrows():
        label_class = label.idxmax()
        attack_type = data_mgr.attack_map[label_class]
        attack_type_index = data_mgr.attack_types.index(attack_type)
        true_attack_type_indices.append(attack_type_index)
    return true_attack_type_indices


def get_cic_true_attack_type_indices(labels: np.ndarray, data_mgr: CICDataManager, one_vs_all: bool) -> list:
    """
    Maps CIC attack names to attack type indices.

    Args:
        labels (pd.Series): Series of attack names (e.g., 'Benign', 'DoS', etc.).
        data_mgr (CICDataManager): The CIC data manager containing attack mappings.

    Returns:
        list: List of attack type indices corresponding to the labels.
    """
    true_attack_type_indices = []  # List of attack type indices
    for label in labels:
        if one_vs_all: # One vs All mapping training on only benign/normal traffic.
            if label == "Benign":
                true_attack_type_indices.append(0)
            else: # All other attacks are considered as attack (1)
                true_attack_type_indices.append(1)
        elif label in data_mgr.attack_map: # Map attack name to attack type
            attack_type = data_mgr.attack_map[label]  # Map attack name to attack type
            attack_type_index = data_mgr.attack_types.index(attack_type)  # Get index of attack type
            true_attack_type_indices.append(attack_type_index)
        else:
            # Handle unmapped labels (e.g., unknown attacks)
            logging.warning(f"Unmapped label: {label}")
            true_attack_type_indices.append(-1)  # Use -1 for unmapped labels
    return true_attack_type_indices

    
def calculate_metrics(true_positive, true_negative, false_positive, false_negative):
    if (true_positive + true_negative + false_positive + false_negative) != 0:
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    else:
        accuracy = 0
    if (true_positive + false_negative) != 0:
        recall = true_positive / (true_positive + false_negative) # a.k.a. true positive rate, sensitivity,  hit rate
    else:
        recall = 0
    if (true_negative + false_positive) != 0:
        specifity = true_negative / (true_negative + false_positive) # a.k.a. true negative rate, selectivity, how precise is the model when it predicts a negative class
    else:
        specifity = 0
    if (true_positive + false_positive) != 0:
        precision = true_positive / (true_positive + false_positive) # how precise is the model when it predicts a positive class
    else:
        precision = 0
    if (recall + precision) != 0:
        f1_score = (2 * recall * precision)/(recall + precision)
    else:
        f1_score = 0
    if (false_positive + true_negative) != 0:
        fpr = false_positive / (false_positive + true_negative) # false positive rate, how many negative samples are incorrectly classified as positive
    else:
        fpr = 0
    return accuracy, recall, specifity, precision, f1_score, fpr
    
""" Example usage:
true_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
predicted_labels = [0, 2, 1, 0, 0, 2, 1, 1, 2]
num_classes = 3

metrics = calculate_tp_tn_fp_fn(true_labels, predicted_labels, num_classes)
for cls, values in metrics.items():
    print(f"Class {cls}: TP={values['TP']}, TN={values['TN']}, FP={values['FP']}, FN={values['FN']}")
"""
def calculate_tp_tn_fp_fn(true_labels, predicted_labels, num_classes):
    """
    Calculates TP, TN, FP, FN for each class in a multi-class classification problem.

    Args:
        true_labels (list or np.ndarray): True class labels.
        predicted_labels (list or np.ndarray): Predicted class labels.
        num_classes (int): Number of classes.

    Returns:
        dict: A dictionary containing TP, TN, FP, FN for each class.
    """
    metrics = {cls: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for cls in range(num_classes)}

    for cls in range(num_classes):
        for true, pred in zip(true_labels, predicted_labels):
            if true == cls and pred == cls:
                metrics[cls]["TP"] += 1  # True Positive
            elif true != cls and pred != cls:
                metrics[cls]["TN"] += 1  # True Negative
            elif true != cls and pred == cls:
                metrics[cls]["FP"] += 1  # False Positive
            elif true == cls and pred != cls:
                metrics[cls]["FN"] += 1  # False Negative

    return metrics