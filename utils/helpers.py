import io
import logging
from typing import List, Tuple
import numpy as np
import os
import pandas as pd
import requests

from datetime import datetime
from models.agent import Agent
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from models.attack_agent import AttackAgent
from models.defender_agent import DefenderAgent
from models.rl_env import RLenv
##################
# Helper Methods #
##################

def get_attack_type_maps(attack_map: dict[str,str], attack_names: list[str]):
    """
    Get distinct attack maps for all four attack types: DoS, Probe, R2L, U2R.
    Given the full attack_map and the attack_names that are present in the used dataset, return the attack maps for each attack type.
    Each attack map is a dictionary where the key is the index of the attack in the full attack_map and the value is the attack name.

    In each attack map, index 0 is reserved for the normal class, as each attacker is expected to be able to behave normally.

    Args:
        attack_map (dict[str,str]): The full attack map.
        attack_names (list[str]): The attack names present in the used dataset.
        
    Returns:
        Tuple[dict[int,str], dict[int,str], dict[int,str], dict[int,str]]: The attack maps for DoS, Probe, R2L, U2R.
    """
    attack_map_dos = {index: attack for index, (attack, attack_type) in enumerate(attack_map.items()) if attack_type == 'DoS' and attack in attack_names}
    attack_map_dos[0] = 'normal'
    attack_map_dos = list(attack_map_dos.keys())
    attack_map_dos.sort()
    attack_map_probe = {index: attack for index, (attack, attack_type) in enumerate(attack_map.items()) if attack_type == 'Probe' and attack in attack_names}
    attack_map_probe[0] = 'normal'
    attack_map_probe = list(attack_map_probe.keys())
    attack_map_probe.sort()
    attack_map_r2l = {index: attack for index, (attack, attack_type) in enumerate(attack_map.items()) if attack_type == 'R2L' and attack in attack_names}
    attack_map_r2l[0] = 'normal'
    attack_map_r2l = list(attack_map_r2l.keys())
    attack_map_r2l.sort()
    attack_map_u2r = {index: attack for index, (attack, attack_type) in enumerate(attack_map.items()) if attack_type == 'U2R' and attack in attack_names}
    attack_map_u2r[0] = 'normal'
    attack_map_u2r = list(attack_map_u2r.keys())
    attack_map_u2r.sort()
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
        logging.info(f"Downloaded: {kdd_train_url}\nSaved in: {kdd_train}")
    if (not os.path.exists(kdd_test)):
        kdd_test_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTest%2B.txt"
        download_file(kdd_test_url, kdd_test)
        logging.info(f"Downloaded: {kdd_test_url}\nSaved in: {kdd_test}")

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

def calculate_f1_scores_per_class(predicted_actions, attack_types, true_labels, **kwargs):
    predicted_actions_dummies = pd.get_dummies(predicted_actions)
    posible_actions = np.arange(len(attack_types))
    for non_existing_action in posible_actions:
        if non_existing_action not in predicted_actions_dummies.columns:
            predicted_actions_dummies[non_existing_action] = np.uint8(0)
    true_labels_dummies = pd.get_dummies(true_labels)

    if kwargs.get('one_vs_all', False):
        normal_f1_score = f1_score(true_labels_dummies[0].values, predicted_actions_dummies[0].values)
        dos_f1_score = f1_score(true_labels_dummies[1].values, predicted_actions_dummies[1].values)
        probe_f1_score = f1_score(true_labels_dummies[2].values, predicted_actions_dummies[2].values)
        r2l_f1_score = f1_score(true_labels_dummies[3].values, predicted_actions_dummies[3].values)
        u2r_f1_score = f1_score(true_labels_dummies[4].values, predicted_actions_dummies[4].values)
        overall_f1_score = f1_score(true_labels, predicted_actions, average='weighted')
        return [normal_f1_score, dos_f1_score, probe_f1_score, r2l_f1_score, u2r_f1_score, overall_f1_score]
    else:
        normal_f1_score = f1_score(true_labels_dummies[0].values, predicted_actions_dummies[0].values)
        attack_f1_score = f1_score(true_labels_dummies[1].values, predicted_actions_dummies[1].values)
        overall_f1_score = f1_score(true_labels, predicted_actions, average='weighted')
        return [normal_f1_score, attack_f1_score, overall_f1_score]

def calculate_f1_scores_per_class_dynamically(predicted_actions, attack_types, true_labels, one_vs_all=False, target_class=None):
    """
    Dynamically calculates F1 scores for each class and optionally supports one-vs-all scenarios.

    Args:
        predicted_actions (list or np.ndarray): Predicted class labels.
        attack_types (list): List of attack types (e.g., ['normal', 'DoS', 'Probe', 'R2L', 'U2R']).
        true_labels (list or np.ndarray): True class labels.
        one_vs_all (bool): Whether to calculate F1 scores in a one-vs-all scenario. Default is False.
        target_class (int): The target class for one-vs-all. Required if one_vs_all=True.

    Returns:
        list: List of F1 scores for each class and the overall weighted F1 score.
    """
    # One-Hot-Encoding der vorhergesagten und tatsächlichen Labels
    predicted_actions_dummies = pd.get_dummies(predicted_actions)
    true_labels_dummies = pd.get_dummies(true_labels)

    # Sicherstellen, dass alle möglichen Aktionen vorhanden sind
    possible_actions = np.arange(len(attack_types))
    for non_existing_action in possible_actions:
        if non_existing_action not in predicted_actions_dummies.columns:
            predicted_actions_dummies[non_existing_action] = np.uint8(0)
        if non_existing_action not in true_labels_dummies.columns:
            true_labels_dummies[non_existing_action] = np.uint8(0)

    # Falls one-vs-all aktiviert ist
    if one_vs_all:
        if target_class is None:
            raise ValueError("For one-vs-all mode, 'target_class' must be specified.")
        
        # Zielklasse als positiv, alle anderen als negativ
        true_binary = (np.array(true_labels) == target_class).astype(int)
        predicted_binary = (np.array(predicted_actions) == target_class).astype(int)
        
        # F1-Score für die Zielklasse berechnen
        f1 = f1_score(true_binary, predicted_binary)
        return [f1]  # Nur ein F1-Score für die Zielklasse

    # Dynamische Berechnung der F1-Scores für alle Klassen
    f1_scores = []
    for action in possible_actions:
        f1 = f1_score(true_labels_dummies[action].values, predicted_actions_dummies[action].values)
        f1_scores.append(f1)

    # Gesamtgewichteter F1-Score
    overall_f1_score = f1_score(true_labels, predicted_actions, average='weighted')
    f1_scores.append(overall_f1_score)

    return f1_scores

def print_aggregated_performance_measures(predicted_actions, true_labels):
    logging.info('Performance measures on Test data')
    logging.info('Accuracy =  {:.4f}'.format(accuracy_score(true_labels, predicted_actions)))
    logging.info('Weighted average F1 =  {:.4f}'.format(f1_score(true_labels, predicted_actions, average='weighted')))
    logging.info('Weighted average Precision_score =  {:.4f}'.format(precision_score(true_labels, predicted_actions, average='weighted')))
    logging.info('Weighted recall_score =  {:.4f}'.format(recall_score(true_labels, predicted_actions, average='weighted')))

def calculate_one_vs_all_metrics(true_attack_type_indices, actions, **kwargs):
    mapa = kwargs.get('attack_type', {0:'normal', 1:'DoS', 2:'Probe', 3:'R2L', 4:'U2R'})
    if isinstance(mapa, list): 
        mapa = {i: label for i, label in enumerate(mapa)}
        
    yt_app = pd.Series(true_attack_type_indices).map(mapa)

    perf_per_class = pd.DataFrame(columns=['name', 'accuracy', 'f1', 'precision', 'recall'])
    #perf_per_class = pd.DataFrame(index=range(len(yt_app.unique())),columns=['name', 'accuracy','f1', 'precision','recall'])
    #for i,x in enumerate(pd.Series(yt_app).value_counts().index):
    for x in yt_app.unique():
        #y_test_hat_check = pd.Series(actions).map(mapa).copy()
        #y_test_hat_check[y_test_hat_check != x] = 'OTHER'
        #yt_app = pd.Series(true_attack_type_indices).map(mapa).copy()
        #yt_app[yt_app != x] = 'OTHER'
        #ac=accuracy_score(yt_app, y_test_hat_check)
        #f1=f1_score(yt_app, y_test_hat_check, pos_label=x, average='binary', zero_division=0)
        #pr=precision_score(yt_app, y_test_hat_check, pos_label=x, average='binary', zero_division=0)
        #re=recall_score(yt_app, y_test_hat_check, pos_label=x, average='binary', zero_division=0)
        #perf_per_class.iloc[i]=[x,ac,f1,pr,re]
        y_pred = pd.Series(actions).map(mapa).copy()
        y_true = yt_app.copy()

        y_pred[y_pred != x] = 'OTHER'
        y_true[y_true != x] = 'OTHER'

        perf_per_class.loc[len(perf_per_class)] = [
            x,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, pos_label=x, average='binary', zero_division=0),
            precision_score(y_true, y_pred, pos_label=x, average='binary', zero_division=0),
            recall_score(y_true, y_pred, pos_label=x, average='binary', zero_division=0)
        ]
        
    return perf_per_class

def calculate_general_overview_per_attack_type(attack_types, estimated_labels, estimated_correct_labels, true_labels, f1_scores, mismatch) -> pd.DataFrame:
    outputs_df = pd.DataFrame(index = attack_types, columns = ["Estimated", "Correct", "Total", "Mismatch", "F1_score"])
    for indx, _ in enumerate(attack_types):
        outputs_df.iloc[indx].Estimated = estimated_labels[indx]
        outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
        outputs_df.iloc[indx].Total = true_labels[indx]
        outputs_df.iloc[indx].Mismatch = abs(mismatch[indx])
        outputs_df.iloc[indx].F1_score = f1_scores[indx]*100

    # Add a row for the general F1 score
    general_f1_score = f1_scores[-1]
    general_row = pd.DataFrame([{
        "Estimated": "",
        "Correct": "",
        "Total": "",
        "Mismatch": "",
        "F1_score": general_f1_score * 100
    }], index=["Weigthed Avg."])

    outputs_df = pd.concat([outputs_df, general_row])

    return outputs_df

def transform_attacks_by_epoch(attacks_by_epoch, attack_id_to_index, num_attack_types):
    """
    Transforms the current attacks_by_epoch structure into the expected format for plotting.

    Args:
        attacks_by_epoch (list): Current nested list structure of attacks_by_epoch.
        attack_id_to_index (dict): Mapping from attack id to index in attack_names.
        num_attack_types (int): Number of attack types.

    Returns:
        list: Transformed structure where each attacker has a list of epochs with attack frequencies.
    """
    num_attackers = len(attacks_by_epoch[0][0])
    transformed = [[] for _ in range(num_attackers)]

    for epoch in attacks_by_epoch:
        epoch_counts = [np.zeros(num_attack_types, dtype=int) for _ in range(num_attackers)]
        for iteration in epoch:
            for attacker_idx, attack in enumerate(iteration):
                attack_index = attack_id_to_index.get(int(attack[0]), -1)
                if attack_index != -1:
                    epoch_counts[attacker_idx][attack_index] += 1
        for attacker_idx in range(num_attackers):
            transformed[attacker_idx].append(epoch_counts[attacker_idx])

    return transformed

def transform_attacks_by_type(attacks_by_epoch, attack_id_to_type, attack_types):
    """
    Transforms the current attacks_by_epoch structure into the expected format for plotting by attack types.

    Args:
        attacks_by_epoch (list): Current nested list structure of attacks_by_epoch.
        attack_id_to_type (dict): Mapping from attack id to attack type.
        attack_types (list): List of attack types.

    Returns:
        list: Transformed structure where each attacker has a list of epochs with attack type frequencies.
    """
    num_attackers = len(attacks_by_epoch[0][0])  # Anzahl der Angreifer
    transformed = [[] for _ in range(num_attackers)]

    for epoch in attacks_by_epoch:
        epoch_counts = [{attack_type: 0 for attack_type in attack_types} for _ in range(num_attackers)]
        for iteration in epoch:
            for attacker_idx, attack in enumerate(iteration):
                attack_type = attack_id_to_type.get(int(attack[0]), 'Unknown')
                if attack_type in epoch_counts[attacker_idx]:
                    epoch_counts[attacker_idx][attack_type] += 1
        for attacker_idx in range(num_attackers):
            transformed[attacker_idx].append([epoch_counts[attacker_idx][attack_type] for attack_type in attack_types])

    return transformed

def create_attack_id_to_index_mapping(attack_map, attack_names) -> dict:
    """
    Creates a mapping from attack id to index of attack_names.

    Args:
        attack_map (dict): Dictionary mapping attack names to attack types.
        attack_names (list): List of attack names that are present in the dataset.

    Returns:
        dict: Mapping from attack id to index in attack_names.
    """
    return {idx: attack_names.index(label) for idx, label in enumerate(attack_map.keys()) if label in attack_names}

def create_attack_id_to_type_mapping(attack_map) -> dict:
    """
    Creates a mapping from attack id to attack type.

    Args:
        attack_map (dict): Dictionary mapping attack names to attack types.

    Returns:
        dict: Mapping from attack id to attack type.
    """
    attack_id_to_type = {}
    for idx, (attack_name, attack_type) in enumerate(attack_map.items()):
        attack_id_to_type[idx] = attack_type
    return attack_id_to_type

def get_attack_actions(attackers: List[AttackAgent], initial_states: pd.DataFrame) -> List[List[int]]:
    """
    Get actions/attacks of all attackers based on their policies.

    Args:
        attackers (list): List of attacker agents.
        initial_states (list): List of initial state for each attacker.

    Returns:
        list: List of lists of attack action(s) for each attacker.
        The order depends on the order of the attackers. (expected: DoS, Probe, R2L, U2R)
    """
    return [attacker.act(initial_states) for attacker in attackers]

def store_experience(agents: List[DefenderAgent | AttackAgent], states: List[pd.DataFrame], actions: List[List[int]], next_states: List[pd.DataFrame], rewards: List[int], done):
    """
    Stores the experience in the memory of the agents.
    For a given state and action, the next state, reward, and done flag are stored.

    Hint: The agent stores its actions and rewards for the given states. 
    After the episode is done, the agent will learn from the stored experiences.

    Args:
        agents (list): List of agents (e.g., attackers or defender).
        states (list): Current states for each agent.
        actions (list): Actions taken by each agent.
        next_states (list): Next states for each agent.
        rewards (list): Rewards received by each agent.
        done (bool): Whether the episode is done.
    """
    if(agents[0].name == "Defender"):
        for state, action, next_state, reward in zip(states, actions, next_states, rewards):
            defender = agents[0]
            defender.learn(state, action, next_state, reward, done)
    else:
        for agent, state, action, next_state, reward in zip(agents, states, actions, next_states, rewards):
            agent.learn(state, action, next_state, reward, done)

def update_models_and_statistics(agent_defender: DefenderAgent, attackers: List[AttackAgent], def_loss, att_loss_dos, att_loss_probe, 
                                 att_loss_r2l, att_loss_u2r, agg_att_loss, def_metrics_chain: List[dict[str, any]], 
                                 att_metrics_chain: List[dict[str,any]], epoch_mse_before: list, epoch_mae_before: list, sample_indices_list: list):
    """
    Updates the models of the defender and attackers and updates the loss and statistics.

    Args:
        agent_defender (DefenderAgent): The defender agent.
        attackers (list): List of attacker agents (DoS, Probe, R2L, U2R).
        def_loss (float): Current cumulative loss for the defender.
        att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r (float): Current cumulative losses for the attackers.
        agg_att_loss (float): Aggregate loss for all attackers.
        def_metrics_chain (list): List to store defender metrics.
        att_metrics_chain (list): List to store attacker metrics.
        epoch_mse_before, epoch_mae_before, epoch_mse_after, epoch_mae_after (list): Lists to store MSE and MAE metrics.

    Returns:
        tuple: Updated values for def_loss, att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, agg_att_loss.
    """
    # Train defender
    def_metrics = agent_defender.update_model()
    def_loss += def_metrics["loss"]

    # Train attackers
    att_metrics_dos = attackers[0].update_model()
    att_metrics_probe = attackers[1].update_model()
    att_metrics_r2l = attackers[2].update_model()
    att_metrics_u2r = attackers[3].update_model()

    # Update attacker losses
    att_loss_dos += att_metrics_dos["loss"]
    att_loss_probe += att_metrics_probe["loss"]
    att_loss_r2l += att_metrics_r2l["loss"]
    att_loss_u2r += att_metrics_u2r["loss"]
    agg_att_loss += att_metrics_dos["loss"] + att_metrics_probe["loss"] + att_metrics_r2l["loss"] + att_metrics_u2r["loss"]

    # Update metrics
    def_metrics_chain.append(def_metrics)
    att_metrics_chain.extend([att_metrics_dos, att_metrics_probe, att_metrics_r2l, att_metrics_u2r])
    epoch_mse_before.append(def_metrics["mse_before"])
    epoch_mae_before.append(def_metrics["mae_before"])

    # Used samples
    sample_indices = def_metrics["sample_indices"]
    sample_indices_list.append(sample_indices)

    return def_loss, att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, agg_att_loss

def update_models_and_statistics_cic(agent_defender: DefenderAgent, attackers: AttackAgent, def_loss, att_loss, def_metrics_chain: List[dict[str, any]], 
                                 att_metrics_chain: List[dict[str,any]], epoch_mse_before: list, epoch_mae_before: list, sample_indices_list: list):
    """
    Updates the models of the defender and attackers and updates the loss and statistics.

    Args:
        agent_defender (DefenderAgent): The defender agent.
        attackers (list): List of attacker agents (DoS, Probe, R2L, U2R).
        def_loss (float): Current cumulative loss for the defender.
        agg_att_loss (float): Aggregate loss for all attackers.
        def_metrics_chain (list): List to store defender metrics.
        att_metrics_chain (list): List to store attacker metrics.
        epoch_mse_before, epoch_mae_before, epoch_mse_after, epoch_mae_after (list): Lists to store MSE and MAE metrics.

    Returns:
        tuple: Updated values for def_loss, att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, agg_att_loss.
    """
    # Train defender
    def_metrics = agent_defender.update_model()
    def_loss += def_metrics["loss"]

    # Train attackers
    att_metrics = attackers.update_model()

    # Update attacker losses
    att_loss += att_metrics["loss"]

    # Update metrics
    def_metrics_chain.append(def_metrics)
    att_metrics_chain.extend([att_metrics])
    epoch_mse_before.append(def_metrics["mse_before"])
    epoch_mae_before.append(def_metrics["mae_before"])

    # Used samples
    sample_indices = def_metrics["sample_indices"]
    sample_indices_list.append(sample_indices)

    return def_loss, att_loss

def update_episode_statistics(def_reward, att_reward, def_total_reward_by_episode, att_total_reward_by_episode,
                              att_total_reward_by_episode_dos, att_total_reward_by_episode_probe,
                              att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r):
    """
    Updates the statistics for the current episode.

    Args:
        def_reward (np.array): Rewards for the defender in the current iteration.
        att_reward (list): Rewards for the attackers in the current iteration.
        def_total_reward_by_episode (int): Total reward for the defender in the current episode.
        att_total_reward_by_episode (int): Total reward for all attackers in the current episode.
        att_total_reward_by_episode_dos (int): Total reward for DoS attacker in the current episode.
        att_total_reward_by_episode_probe (int): Total reward for Probe attacker in the current episode.
        att_total_reward_by_episode_r2l (int): Total reward for R2L attacker in the current episode.
        att_total_reward_by_episode_u2r (int): Total reward for U2R attacker in the current episode.

    Returns:
        Updated statistics for the episode.
    """
    def_total_reward_by_episode += np.sum(def_reward, dtype=np.int32)
    att_total_reward_by_episode += sum(np.sum(reward, dtype=np.int32) for reward in att_reward)
    att_total_reward_by_episode_dos += np.sum(att_reward[0], dtype=np.int32)
    att_total_reward_by_episode_probe += np.sum(att_reward[1], dtype=np.int32)
    att_total_reward_by_episode_r2l += np.sum(att_reward[2], dtype=np.int32)
    att_total_reward_by_episode_u2r += np.sum(att_reward[3], dtype=np.int32)

    return (def_total_reward_by_episode, att_total_reward_by_episode, att_total_reward_by_episode_dos,
            att_total_reward_by_episode_probe, att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r)

def update_episode_statistics_cic(def_reward, att_reward, def_total_reward_by_episode, att_total_reward_by_episode):
    """
    Updates the statistics for the current episode.

    Args:
        def_reward (np.array): Rewards for the defender in the current iteration.
        att_reward (list): Rewards for the attackers in the current iteration.
        def_total_reward_by_episode (int): Total reward for the defender in the current episode.
        att_total_reward_by_episode (int): Total reward for all attackers in the current episode.

    Returns:
        Updated statistics for the episode.
    """
    def_total_reward_by_episode += np.sum(def_reward, dtype=np.int32)
    att_total_reward_by_episode += sum(np.sum(reward, dtype=np.int32) for reward in att_reward)

    return (def_total_reward_by_episode, att_total_reward_by_episode)


def store_episode_results(attack_indices_list, attack_names_list, env: RLenv, epoch_mse_before, epoch_mae_before,
                          def_total_reward_by_episode, att_total_reward_by_episode,
                          att_total_reward_by_episode_dos, att_total_reward_by_episode_probe,
                          att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r, def_loss, agg_att_loss,
                          att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, attack_indices_per_episode: list,
                          attack_names_per_episode: list, attacks_mapped_to_att_type_list: list, mse_before_history: list,
                          mae_before_history: list, def_reward_chain: list,
                          att_reward_chain: list, att_reward_chain_dos: list, att_reward_chain_probe: list, att_reward_chain_r2l: list,
                          att_reward_chain_u2r: list, def_loss_chain: list, att_loss_chain: list, att_loss_chain_dos: list,
                          att_loss_chain_probe: list, att_loss_chain_r2l: list, att_loss_chain_u2r: list, sample_indices_per_episode:list, sample_indices: list):
    """
    Stores the results of the current episode.

    Args:
        (All arguments are lists or variables that store episode results.)

    Returns:
        None
    """
    attack_indices_per_episode.append(attack_indices_list)
    attack_names_per_episode.append(attack_names_list)
    attacks_mapped_to_att_type_list.append(env.att_true_labels)
    if epoch_mse_before:
        mse_before_history.append(np.mean(epoch_mse_before))
        mae_before_history.append(np.mean(epoch_mae_before))

    def_reward_chain.append(def_total_reward_by_episode)
    att_reward_chain.append(att_total_reward_by_episode)
    att_reward_chain_dos.append(att_total_reward_by_episode_dos)
    att_reward_chain_probe.append(att_total_reward_by_episode_probe)
    att_reward_chain_r2l.append(att_total_reward_by_episode_r2l)
    att_reward_chain_u2r.append(att_total_reward_by_episode_u2r)

    def_loss_chain.append(def_loss)
    att_loss_chain.append(agg_att_loss)
    att_loss_chain_dos.append(att_loss_dos)
    att_loss_chain_probe.append(att_loss_probe)
    att_loss_chain_r2l.append(att_loss_r2l)
    att_loss_chain_u2r.append(att_loss_u2r)

    sample_indices_per_episode.append(sample_indices)

def store_episode_results_cic(attack_indices_list, attack_names_list, env: RLenv, epoch_mse_before, epoch_mae_before,
                          def_total_reward_by_episode, att_total_reward_by_episode, def_loss, att_loss,
                          attack_indices_per_episode: list, attack_names_per_episode: list, attacks_mapped_to_att_type_list: list,
                          mse_before_history: list, mae_before_history: list, def_reward_chain: list,
                          att_reward_chain: list, def_loss_chain: list, att_loss_chain: list, sample_indices_per_episode:list, sample_indices: list):
    """
    Stores the results of the current episode.

    Args:
        (All arguments are lists or variables that store episode results.)

    Returns:
        None
    """
    attack_indices_per_episode.append(attack_indices_list)
    attack_names_per_episode.append(attack_names_list)
    attacks_mapped_to_att_type_list.append(env.att_true_labels)
    if epoch_mse_before:
        mse_before_history.append(np.mean(epoch_mse_before))
        mae_before_history.append(np.mean(epoch_mae_before))

    def_reward_chain.append(def_total_reward_by_episode)
    att_reward_chain.append(att_total_reward_by_episode)

    def_loss_chain.append(def_loss)
    att_loss_chain.append(att_loss)

    sample_indices_per_episode.append(sample_indices)

def save_trained_models(agents, output_root_dir, trained_models_dir):
    """
    Saves the trained models for the attackers and the defender.

    Args:
        agents (dict): A dictionary containing the agents to save. Keys should be agent names (e.g., "dos", "probe").
        output_root_dir (str): The root directory for saving the models.
        trained_models_dir (str): The base directory where trained models are stored.

    Returns:
        None
    """
    model_paths = {
        "dos": os.path.join(trained_models_dir, f"{output_root_dir}/attacker_model_dos.keras"),
        "probe": os.path.join(trained_models_dir, f"{output_root_dir}/attacker_model_probe.keras"),
        "r2l": os.path.join(trained_models_dir, f"{output_root_dir}/attacker_model_r2l.keras"),
        "u2r": os.path.join(trained_models_dir, f"{output_root_dir}/attacker_model_u2r.keras"),
        "defender": os.path.join(trained_models_dir, f"{output_root_dir}/defender_model.keras"),
    }

    # Save each model
    save_model(agents["dos"], model_paths["dos"])
    save_model(agents["probe"], model_paths["probe"])
    save_model(agents["r2l"], model_paths["r2l"])
    save_model(agents["u2r"], model_paths["u2r"])
    save_model(agents["defender"], model_paths["defender"])

    logging.info("Saved trained models.")

def save_trained_models_cic(agents, output_root_dir, trained_models_dir):
    """
    Saves the trained models for the attackers and the defender.

    Args:
        agents (dict): A dictionary containing the agents to save. Keys should be agent names (e.g., "dos", "probe").
        output_root_dir (str): The root directory for saving the models.
        trained_models_dir (str): The base directory where trained models are stored.

    Returns:
        None
    """
    model_paths = {
        "attacker": os.path.join(trained_models_dir, f"{output_root_dir}/attacker_model_cic.keras"),
        "defender": os.path.join(trained_models_dir, f"{output_root_dir}/defender_model.keras"),
    }

    # Save each model
    save_model(agents["attacker"], model_paths["attacker"])
    save_model(agents["defender"], model_paths["defender"])

    logging.info("Saved trained models.")