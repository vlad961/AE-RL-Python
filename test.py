import logging
import numpy as np

import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import time

from data.data_cls import DataCls
from models.helpers import calculate_general_overview_per_attack_type, calculate_one_vs_all_metrics, calculate_unique_f1_scores_per_class, get_cf_matrix, get_model_summary, plot_confusion_matrix, print_aggregated_performance_measures

def test_trained_agent_quality(path_to_model, plots_path):
    logging.info("Start testing the trained model.")
    model = tf.keras.models.load_model(path_to_model)
    logging.info(f"Model '{model.name}' loaded from: {path_to_model}")
    logging.info(f"Model summary:\n{get_model_summary(model)}")

    # Define environment, game, make sure the batch_size is the same in train
    test_data = DataCls(dataset_type='test')

    total_reward = 0
    true_labels = np.zeros(len(test_data.attack_types),dtype=int)
    predicted_labels = np.zeros(len(test_data.attack_types),dtype=int)
    predicted_correct_labels = np.zeros(len(test_data.attack_types),dtype=int)

    #states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
    states, labels = test_data.get_full() # get test data and true labels.

    start_time=time.time()
    q = model.predict(states)
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

    f1_scores = calculate_unique_f1_scores_per_class(actions, test_data.attack_types, true_attack_type_indices)
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
    loss, acc_model, precision, recall = model.evaluate(states, pd.get_dummies(true_attack_type_indices), verbose=2)
    logging.info(f"Model metrics: \nloss={loss}, accuracy={acc_model}, precision={precision}, recall={recall}")
    logging.info(f"Time needed for testing: {time.time() - start_time}")