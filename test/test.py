import logging
import numpy as np
import os, sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sklearn.metrics import auc, classification_report, roc_curve
import tensorflow as tf
import time

from data.nsl_kdd_data_manager import NslKddDataManager
from utils.helpers import calculate_general_overview_per_attack_type, calculate_one_vs_all_metrics, calculate_f1_scores_per_class, get_cf_matrix, get_model_summary, print_aggregated_performance_measures
from utils.plotting import plot_confusion_matrix, plot_roc_curve

def test_trained_agent_quality(path_to_model, plots_path):
    logging.info("Start testing the trained model.")
    model = tf.keras.models.load_model(path_to_model)
    logging.info(f"Model '{model.name}' loaded from: {path_to_model}")
    logging.info(f"Model summary:\n{get_model_summary(model)}")

    # Define environment, game, make sure the batch_size is the same in train
    test_data = NslKddDataManager(dataset_type='test')

    total_reward = 0
    true_labels = np.zeros(len(test_data.attack_types),dtype=int)
    predicted_labels = np.zeros(len(test_data.attack_types),dtype=int)
    predicted_correct_labels = np.zeros(len(test_data.attack_types),dtype=int)

    #states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
    states, labels = test_data.get_full() # get test data and true labels.

    start_time=time.time()
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    q = model.predict(states_tensor)
    actions = np.argmax(q, axis=1) # get the action with the highest Q-value -> the predicted attack type

    true_attack_type_indices=[] # list of true attack types as indices (0-4). 0=normal, 1=dos, 2=probe, 3=r2l, 4=u2r. Length = number of samples
    for _, label in labels.iterrows():
        label_class = label.idxmax()
        attack_type = test_data.attack_map[label_class]
        attack_type_index = test_data.attack_types.index(attack_type)
        true_attack_type_indices.append(attack_type_index)

    labels_per_attack_type, counts = np.unique(true_attack_type_indices, return_counts=True)
    true_labels[labels_per_attack_type] += counts

    for indx, a in enumerate(actions):
        predicted_labels[a] +=1
        if a == true_attack_type_indices[indx]:
            total_reward += 1
            predicted_correct_labels[a] += 1

    f1_scores = calculate_f1_scores_per_class(actions, test_data.attack_types, true_attack_type_indices)
    mismatch = predicted_labels - true_labels
    acc = float(100 * total_reward / len(states))

    logging.info(f"Overall overview\nTotal reward: {total_reward} | Number of samples: {len(states)} | Accuracy = {acc:.2f}%")
    
    outputs_df = calculate_general_overview_per_attack_type(test_data.attack_types, predicted_labels, predicted_correct_labels, true_labels, f1_scores, mismatch)
    logging.info(f"Overall overview per attack type\n{outputs_df}")

    aggregated_data_test = np.array(true_attack_type_indices)
    print_aggregated_performance_measures(actions, aggregated_data_test)

    np.set_printoptions(precision=2)
    plot_confusion_matrix(get_cf_matrix(aggregated_data_test, actions), classes=test_data.attack_types, path=plots_path, normalize=True,
                        title='Normalized confusion matrix')

    perf_per_class = calculate_one_vs_all_metrics(true_attack_type_indices, actions)
    logging.info(f"\r\nOne vs All metrics: \r\n{perf_per_class}")
    loss, acc_model, precision, recall, auc_value = model.evaluate(states_tensor, pd.get_dummies(true_attack_type_indices), verbose=2)
    logging.info(f"AUC: {auc_value}")
    # Berechnen der ROC-Kurve und der AUC
    fpr, tpr, _ = roc_curve(pd.get_dummies(true_attack_type_indices).values.ravel(), q.ravel())
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, plots_path)
    logging.info(f"Model metrics: \nloss={loss}, accuracy={acc_model}, precision={precision}, recall={recall}")
    logging.info(f"Optimizer config: {model.optimizer.get_config()}")
    logging.info(f"Time needed for testing: {time.time() - start_time}")
    report = classification_report(true_attack_type_indices, actions, target_names=test_data.attack_types)
    logging.info(f"Classification report:\n{report}")