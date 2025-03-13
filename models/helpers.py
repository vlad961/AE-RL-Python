import io
import logging
import numpy as np
import os
import pandas as pd
import requests

from datetime import datetime
from models.agent import Agent
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
##################
# Helper Methods #
##################

def getAttackTypeMaps(attack_map: dict[str,str], attack_names: list[str]):
    """
    Get single attack maps for all four attack types: DoS, Probe, R2L, U2R.
    Given the full attack_map and the attack_names that are present in the used dataset, return the attack maps for each attack type.
    Each attack map is a dictionary where the key is the index of the attack in the full attack_map and the value is the attack name.

    In each attack map, index 0 is reserved for the normal class.

    Args:
        attack_map (dict[str,str]): The full attack map.
        attack_names (list[str]): The attack names present in the used dataset.
        
    Returns:
        Tuple[dict[int,str], dict[int,str], dict[int,str], dict[int,str]]: The attack maps for DoS, Probe, R2L, U2R.
    """
    attack_map_dos = {index: attack for index, (attack, attack_type) in enumerate(attack_map.items()) if attack_type == 'DoS' and attack in attack_names}
    attack_map_dos[0] = 'normal'
    attack_map_probe = {index: attack for index, (attack, attack_type) in enumerate(attack_map.items()) if attack_type == 'Probe' and attack in attack_names}
    attack_map_probe[0] = 'normal'
    attack_map_r2l = {index: attack for index, (attack, attack_type) in enumerate(attack_map.items()) if attack_type == 'R2L' and attack in attack_names}
    attack_map_r2l[0] = 'normal'
    attack_map_u2r = {index: attack for index, (attack, attack_type) in enumerate(attack_map.items()) if attack_type == 'U2R' and attack in attack_names}
    attack_map_u2r[0] = 'normal'
    return attack_map_dos, attack_map_probe, attack_map_r2l, attack_map_u2r

def download_file(url:str, local_filename:str):
    """
    Download a file from a given URL and save it locally.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The local path where the file will be saved.

    Returns:
        str: The local path where the file was saved.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def save_model(agent: Agent, model_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    # Get the model summary and save it to a file
    agent.model_network.model.save(model_path)
    logging.info(f"Model '{agent.model_network.model_name}' saved in: {model_path}")
    logging.info(f"Model summary:\n{get_model_summary(agent)}")

def get_model_summary(model):
    stream = io.StringIO()
    if isinstance(model, Agent):
        model.model_network.model.summary(print_fn=lambda x: stream.write(x + "\n"))
    else:
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    return summary_str

def download_datasets_if_missing(kdd_train:str, kdd_test:str):
    # If the data files for some reason do not exist, download them from the repo this work is based on.
    if (not os.path.exists(kdd_train)):
        kdd_train_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTrain%2B.txt"
        download_file(kdd_train_url, kdd_train)
        logging.info("Downloaded: {}\nSaved in: {}", kdd_train_url, kdd_train)
    if (not os.path.exists(kdd_test)):
        kdd_test_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTest%2B.txt"
        download_file(kdd_test_url, kdd_test)
        logging.info("Downloaded: {}\nSaved in: {}", kdd_test_url, kdd_test)

def print_total_runtime(script_start_time):
    total_runtime = datetime.now() - script_start_time
    # Convert total runtime to hours, minutes, and seconds
    total_seconds = int(total_runtime.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Total runtime: {hours:02}:{minutes:02}:{seconds:02}")
    logging.info(f"End of the script at: {datetime.now()}")

#####################
# Metrics functions #
#####################
def get_cf_matrix(true_labels, predicted_labels):
    cnf_matrix = confusion_matrix(true_labels, predicted_labels)
    return cnf_matrix

def calculate_f1_scores_per_class(predicted_actions, attack_types, true_labels):
    predicted_actions_dummies = pd.get_dummies(predicted_actions)
    posible_actions = np.arange(len(attack_types))
    for non_existing_action in posible_actions:
        if non_existing_action not in predicted_actions_dummies.columns:
            predicted_actions_dummies[non_existing_action] = np.uint8(0)
    true_labels_dummies = pd.get_dummies(true_labels)

    normal_f1_score = f1_score(true_labels_dummies[0].values, predicted_actions_dummies[0].values)
    dos_f1_score = f1_score(true_labels_dummies[1].values, predicted_actions_dummies[1].values)
    probe_f1_score = f1_score(true_labels_dummies[2].values, predicted_actions_dummies[2].values)
    r2l_f1_score = f1_score(true_labels_dummies[3].values, predicted_actions_dummies[3].values)
    u2r_f1_score = f1_score(true_labels_dummies[4].values, predicted_actions_dummies[4].values)
    overall_f1_score = f1_score(true_labels, predicted_actions, average='weighted')

    return [normal_f1_score, dos_f1_score, probe_f1_score, r2l_f1_score, u2r_f1_score, overall_f1_score]

def print_aggregated_performance_measures(predicted_actions, true_labels):
    logging.info('Performance measures on Test data')
    logging.info('Accuracy =  {:.4f}'.format(accuracy_score(true_labels, predicted_actions)))
    logging.info('F1 =  {:.4f}'.format(f1_score(true_labels, predicted_actions, average='weighted')))
    logging.info('Precision_score =  {:.4f}'.format(precision_score(true_labels, predicted_actions, average='weighted')))
    logging.info('recall_score =  {:.4f}'.format(recall_score(true_labels, predicted_actions, average='weighted')))

def calculate_one_vs_all_metrics(true_attack_type_indices, actions):
    mapa = {0:'normal', 1:'DoS', 2:'Probe',3:'R2L',4:'U2R'}
    yt_app = pd.Series(true_attack_type_indices).map(mapa)

    perf_per_class = pd.DataFrame(index=range(len(yt_app.unique())),columns=['name', 'accuracy','f1', 'precision','recall'])
    for i,x in enumerate(pd.Series(yt_app).value_counts().index):
        y_test_hat_check = pd.Series(actions).map(mapa).copy()
        y_test_hat_check[y_test_hat_check != x] = 'OTHER'
        yt_app = pd.Series(true_attack_type_indices).map(mapa).copy()
        yt_app[yt_app != x] = 'OTHER'
        ac=accuracy_score(yt_app, y_test_hat_check)
        f1=f1_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
        pr=precision_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
        re=recall_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
        perf_per_class.iloc[i]=[x,ac,f1,pr,re]
        
    return perf_per_class

def calculate_general_overview_per_attack_type(attack_types, estimated_labels, estimated_correct_labels, true_labels, f1_scores, mismatch) -> pd.DataFrame:
    outputs_df = pd.DataFrame(index = attack_types, columns = ["Estimated", "Correct", "Total", "F1_score", "Mismatch"])
    for indx, _ in enumerate(attack_types):
        outputs_df.iloc[indx].Estimated = estimated_labels[indx]
        outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
        outputs_df.iloc[indx].Total = true_labels[indx]
        outputs_df.iloc[indx].F1_score = f1_scores[indx]*100
        outputs_df.iloc[indx].Mismatch = abs(mismatch[indx])

    # Add a row for the general F1 score
    general_f1_score = f1_scores[-1]
    general_row = pd.DataFrame([{
        "Estimated": "",
        "Correct": "",
        "Total": "",
        "F1_score": general_f1_score * 100,
        "Mismatch": ""
    }], index=["General"])

    outputs_df = pd.concat([outputs_df, general_row])

    return outputs_df