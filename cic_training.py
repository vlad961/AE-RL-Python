import numpy as np
from data.cic_data_manager import CICDataManager
from utils.config import CICIDS_2017_CLEAN_ALL_BENIGN, CICIDS_2017_CLEAN_ALL_MALICIOUS, CWD, TRAINED_MODELS_DIR
from utils.log_config import log_training_parameters, print_end_of_epoch_info, move_log_files, logger_setup, print_end_of_epoch_info_cic, save_debug_info
from utils.helpers import create_attack_id_to_index_mapping, create_attack_id_to_type_mapping, get_attack_actions, get_attack_states, get_defender_actions, get_attack_type_maps, print_total_runtime, save_trained_models, save_trained_models_cic, store_episode_results, store_episode_results_cic, store_experience, transform_attacks_by_epoch, transform_attacks_by_type, update_episode_statistics, update_episode_statistics_cic, update_models_and_statistics, update_models_and_statistics_cic
from models.rl_env import RLenv
from models.defender_agent import DefenderAgent
from models.attack_agent import AttackAgent
from data.cic_data_manager import cic_attack_map_one_vs_all
from datetime import datetime
from test.test_multiple_agents import test_trained_agent_quality
from utils.plotting_multiple_agents import plot_attack_distribution_for_each_attacker, plot_attack_distributions_multiple_agents, plot_mapped_attack_distribution_for_each_attacker, plot_rewards_and_losses_during_training_multiple_agents, plot_rewards_losses_boxplot, plot_training_error, plot_trend_lines_multiple_agents
import logging
import time
import os, sys
import tensorflow as tf

import gc

"""
This script is the main entry point for the project. It is responsible for downloading the data, training the agents and saving the trained models.

Notes and credits:
The following anomaly detection RL system is based on the work of Guillermo Caminero, Manuel Lopez-Martin, Belen Carro "Adversarial environment reinforcement learning algorithm for intrusion detection".
The original project can be found at: https://github.com/gcamfer/Anomaly-ReactionRL
To be more specific, the code is based on the following file: 'NSL-KDD adaption: AE_RL_NSL-KDD.ipynb' https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/Notebooks/AE_RL_NSL_KDD.ipynb
"""


def main(attack_type=None, file_name_suffix=""):
    # TensorFlow GPU configuration avoids: "W tensorflow/core/data/root_dataset.cc:167] Optimization loop failed: Cancelled: Operation was cancelled" Errors
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            #tf.debugging.set_log_device_placement(True)
    except tf.errors.InvalidArgumentError:
        print("Invalid device or cannot modify virtual devices once initialized.")
        print("Exiting the script early...")
        sys.exit(0)  # Exit with code 0 (success)
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_root_dir = f"{timestamp_begin}{file_name_suffix}"
    plots_path = os.path.join(TRAINED_MODELS_DIR, f"{output_root_dir}/plots/")
    current_log_path = os.path.join(CWD, f"logs/{output_root_dir}.log")
    destination_log_path = os.path.join(TRAINED_MODELS_DIR, f"{output_root_dir}/logs/")
    
    logger_setup(timestamp_begin, file_name_suffix)
    logging.info(f"Started script at: {timestamp_begin}\nCurrent working dir: {CWD}\nUsed data files: \n{CICIDS_2017_CLEAN_ALL_BENIGN},\n{CICIDS_2017_CLEAN_ALL_MALICIOUS}")
    script_start_time = datetime.now()

    try:
        batch_size = 1 # Train batch
        minibatch_size = 100 # batch of memory ExpRep
        experience_replay = True
        iterations_episode = 100
        num_episodes = 3
 
        logging.info("Setting up Attacker and Defender Agents...")
        # Load the CICIDS 2017 dataset
        cic_2017 = True
        data_mgr = CICDataManager(benign_path=CICIDS_2017_CLEAN_ALL_BENIGN,
                                    malicious_path=CICIDS_2017_CLEAN_ALL_MALICIOUS, cic_2017=True, one_vs_all=True)
        attack_valid_actions = list(range(len([0]))) if cic_2017 else list(range(len(data_mgr.attack_names))) # FIXME: use same condition as defender_valid_actions If i allow multiple attacks during training (the training set needs to be adjusted too then)

        # Empirical experience shows that a greater exploration rate is better for the attacker agent.
        att_epsilon = 1
        att_min_epsilon = 0.82  # min value for exploration
        att_gamma = 0.001
        att_decay_rate = 0.99
        att_hidden_size = 100
        att_hidden_layers = 5
        att_learning_rate = 0.001

        # Defender is trained to detect type of attacks 0: attack, 1: normal FIXME: hier evtl attack und noraml wechseln
        defender_valid_actions = list(range(len(data_mgr.attack_types)))
        defender_num_actions = len(defender_valid_actions)
        def_epsilon = 1  # exploration
        def_min_epsilon = 0.01  # min value for exploration
        def_gamma = 0.001
        def_decay_rate = 0.99
        def_hidden_size = 100
        def_hidden_layers = 3
        def_learning_rate = 0.001

        training_params = {"num_episodes": num_episodes, "iterations_episode": iterations_episode, "minibatch_size": minibatch_size,
                           "total_samples": num_episodes * iterations_episode, "data_shape": data_mgr.shape}
        attacker_params = {"num_actions": len(attack_valid_actions),"gamma": att_gamma, "epsilon": att_epsilon, "hidden_size": att_hidden_size,
                           "hidden_layers": att_hidden_layers, "learning_rate": att_learning_rate }
        defender_params = {"num_actions": defender_num_actions, "gamma": def_gamma, "epsilon": def_epsilon, "hidden_size": def_hidden_size,
                           "hidden_layers": def_hidden_layers,"learning_rate": def_learning_rate}

        # Create the attacker and defender agents
        agent_cic_2017_attacker = AttackAgent(attack_valid_actions, data_mgr.obs_size, "CIC-2017", "EpsilonGreedy",
                                        epoch_length=iterations_episode,
                                        epsilon=att_epsilon,
                                        min_epsilon=att_min_epsilon,
                                        decay_rate=att_decay_rate,
                                        gamma=att_gamma,
                                        hidden_size=att_hidden_size,
                                        hidden_layers=att_hidden_layers,
                                        minibatch_size=minibatch_size,
                                        mem_size=1000,
                                        learning_rate=att_learning_rate,
                                        ExpRep=experience_replay,
                                        target_model_name='attacker_target_model_cic_2017',
                                        model_name='attacker_model_cic_2017',
                                        multiple_attacker=False,
                                        amount_attackers=1,
                                        )
        
        attackers = [agent_cic_2017_attacker]

        agent_defender = DefenderAgent(defender_valid_actions, data_mgr.obs_size, "EpsilonGreedy",
                                        epoch_length=iterations_episode,
                                        epsilon=def_epsilon,
                                        min_epsilon=def_min_epsilon,
                                        decay_rate=def_decay_rate,
                                        gamma=def_gamma,
                                        hidden_size=def_hidden_size,
                                        hidden_layers=def_hidden_layers,
                                        minibatch_size=100, # //TODO: höhere minibatch_size für schnelleres Training
                                        mem_size=1000,
                                        learning_rate=def_learning_rate,
                                        ExpRep=experience_replay,
                                        target_model_name='defender_target_model',
                                        model_name='defender_model',
                                        multiple_attacker=False,
                                        amount_attackers=1,
                                        )

        # Create the environment
        logging.info("Creating environment...")
        env = RLenv(data_mgr, attackers, agent_defender, batch_size=batch_size, iterations_episode=iterations_episode)

        # Print training parameters
        if cic_2017:
            attack_names = ['Benign']
        else:
            attack_names = data_mgr.attack_names
        log_training_parameters(training_params, attacker_params, defender_params, attack_type, attack_names)
        # Statistics
        att_reward_chain, def_reward_chain, att_loss_chain, def_loss_chain = [], [], [], []
        def_metrics_chain, att_metrics_chain = [], []
        mse_before_history, mae_before_history = [], []
        
        # Main loops
        attack_indices_per_episode, attack_names_per_episode = [], []
        attacks_mapped_to_att_type_list = []
        sample_indices_per_episode = []

        for episode in range(num_episodes):
            epoch_start_time = time.time()
            epoch_mse_before, epoch_mae_before = [], []

            # Attack and defense losses
            def_loss, att_loss = 0.0, 0.0
            # Total rewards by episode / Total rewards by episode for each attack type
            def_total_reward_by_episode, att_total_reward_by_episode = 0, 0
            
            attack_indices_list = []
            attack_names_list = []
            sample_indices_list = []
            done = False
            
            # Reset the environment and initialize a new batch of random states and corresponding attack labels.
            initial_state, labels = env.reset()
            # Determine the attack actions for the randomly chosen initial states based on the attackers' policies.
            # Depending on the epsilon value, the attackers either exploit their learned policy to predict the best actions
            # or explore random actions (Exploitation vs. Exploration).
            attack_actions = get_attack_actions(attackers, initial_state)

            # Retrieve the next states based on the chosen attack actions of the attacker agents.
            # Each state represents the environment after the execution of the corresponding attack.
            # The states are derived from the IDS dataset used in the environment.
            states, labels, labels_names = get_attack_states(env, attack_actions)

            # Iteration in one episode
            for iteration in range(iterations_episode):
                attack_indices_list.append(attack_actions)
                attack_names_list.append(labels_names)
                # Determine the defender agent's actions/classifications for the given attack states.
                defender_actions = get_defender_actions(agent_defender, states)

                # Enviroment actuation for those actions
                next_states, next_labels, next_labels_names, def_reward, att_reward, next_attack_actions, done = env.act(defender_actions,
                                                                                            attack_actions, states)

                # Store experiences
                store_experience(attackers, states, attack_actions, next_states, att_reward, done)
                store_experience([agent_defender], states, defender_actions, next_states, def_reward, done)

                # Train network, update loss after at least minibatch_size (observations)
                if experience_replay and episode * iterations_episode + iteration >= minibatch_size:
                    
                    def_loss, att_loss = update_models_and_statistics_cic(agent_defender, attackers[0], def_loss, att_loss,
                            def_metrics_chain, att_metrics_chain, epoch_mse_before, epoch_mae_before, sample_indices_list)

                # Update the environment for the next iteration
                states = next_states
                attack_actions = next_attack_actions
                # Update the labels for the next iteration used only for debugging reasons for a better understanding of the agent's behavior
                labels_names = next_labels_names

                # Update episode statistics
                (def_total_reward_by_episode, att_total_reward_by_episode) = update_episode_statistics_cic(def_reward, att_reward, 
                            def_total_reward_by_episode, att_total_reward_by_episode)

            # Store episode results
            store_episode_results_cic(attack_indices_list, attack_names_list, env, epoch_mse_before, epoch_mae_before,
                                def_total_reward_by_episode, att_total_reward_by_episode, def_loss, att_loss,
                                attack_indices_per_episode, attack_names_per_episode, attacks_mapped_to_att_type_list,
                                mse_before_history, mae_before_history, def_reward_chain, att_reward_chain, def_loss_chain,
                                att_loss_chain, sample_indices_per_episode, sample_indices_list)

            end_time = time.time()
            # Update user view
            episode_info = {"episode": episode, "num_episodes": num_episodes, "epoch_start_time": epoch_start_time, "end_time": end_time}
            metrics = {"def_loss": def_loss, "def_total_reward_by_episode": def_total_reward_by_episode,
                       "att_loss": att_loss, "att_total_reward_by_episode": att_total_reward_by_episode}

            print_end_of_epoch_info_cic(episode_info, metrics, env)
            gc.collect()

        # Save the trained models
        agents = {"attacker": agent_cic_2017_attacker, "defender": agent_defender}
        save_trained_models_cic(agents, output_root_dir, TRAINED_MODELS_DIR)
        print_total_runtime(script_start_time)

        #################################
        # Collect all relevant variables#
        #################################
        debug_info = {
            "rewards": {
                "defender": def_reward_chain,
                "aggregated_attacker": att_reward_chain,
            },
            "losses": {
                "defender": def_loss_chain,
                "aggregated_attacker": att_loss_chain,
            },
            "attack_indices_per_episode": attack_indices_per_episode,
            "attack_names_per_episode": attack_names_per_episode,
            "attacks_mapped_to_att_type_list": attacks_mapped_to_att_type_list,
            "mse_before_history": mse_before_history,
            "mae_before_history": mae_before_history,
            "agents": {
                "defender": agent_defender.name,
                "attacker": agent_cic_2017_attacker.name,
            },
            "sample_indices_per_episode": sample_indices_per_episode,
            "attack_names": data_mgr.attack_names,
            "attack_types": data_mgr.attack_types,
            "plots_path": plots_path,
            "output_root_dir": output_root_dir,
            "destination_log_path": destination_log_path,
        }
        save_debug_info(destination_log_path, **debug_info)

        #############################
        # Test and visualize results#
        #############################
        logging.info(f"Trying to save summary plots under: {plots_path}")
        plot_rewards_and_losses_during_training_multiple_agents(def_reward_chain, att_reward_chain, def_loss_chain, att_loss_chain, plots_path)
        plot_attack_distributions_multiple_agents(attack_indices_per_episode, cic_attack_map_one_vs_all, data_mgr.attack_names, attacks_mapped_to_att_type_list, plots_path)
        rewards = [def_reward_chain, att_reward_chain]
        losses = [def_loss_chain, att_loss_chain]
        plot_trend_lines_multiple_agents(rewards, losses, [agent_defender.name, agent_cic_2017_attacker.name], plots_path)

        attack_id_to_index = create_attack_id_to_index_mapping(cic_attack_map_one_vs_all, data_mgr.attack_names)
        num_attacks = len(data_mgr.attack_names)
        transformed_attacks = transform_attacks_by_epoch(attack_indices_per_episode, attack_id_to_index, num_attacks)
        plot_attack_distribution_for_each_attacker(transformed_attacks, data_mgr.attack_names, plots_path, ['Attacker CIC-2017'])
        
        attack_id_to_type = create_attack_id_to_type_mapping(cic_attack_map_one_vs_all)
        # Daten transformieren
        transformed_attacks_by_type = transform_attacks_by_type(attack_indices_per_episode, attack_id_to_type, data_mgr.attack_types)

        plot_mapped_attack_distribution_for_each_attacker(transformed_attacks_by_type, ['Benign','Malicious'], plots_path, ['Attacker CIC-2017'])
        plot_rewards_losses_boxplot(rewards, losses, [agent_defender.name, agent_cic_2017_attacker.name], plots_path)# FIXME: überprüfe ob Logik korrekt ist. 
        # Nutze die Funktion nach Abschluss deines Trainings:
        plot_training_error(mse_before_history, mae_before_history, save_path=plots_path)
        
        defender_model_path = os.path.join(TRAINED_MODELS_DIR, f"{output_root_dir}/defender_model.keras")
        test_trained_agent_quality(defender_model_path, plots_path)
        move_log_files(current_log_path, destination_log_path)
    except Exception as e:
        logging.error(f"Error occurred\n:{e}", exc_info=True)

# Run the main function
if __name__ == "__main__":
    main(file_name_suffix="-WIN-multiple-attackers-formated-data-att-5L-def-3L-lr-0.001")
    #main("U2R", file_name_suffix="-WIN-only-DoS")
    #main("equally_balanced_data", file_name_suffix="-WIN-equally-balanced-data")
    #main(["normal", "R2L", "U2R"], file_name_suffix="-WIN-normal-r2l-u2r-attacks-att-5L-def-3L-lr-0.001") # Run the main function with a list of specific attack types (normal, DoS, Probe, R2L, U2R)
    #main(["normal", "U2R"], file_name_suffix="-WIN-normal-U2R") # Run the main function with a list of specific attack types (normal, DoS, Probe, R2L, U2R)
    #main() # Run the main function with all attack types (normal, DoS, Probe, R2L, U2R) and save the results in a default folder (timestamp).
    #main(file_name_suffix="mac-all-attacks") # Run the main function with all attack types and save the results in a specific folder (mac-all-attacks)
    #main("normal") # Run the main function with a specific attack type (normal, DoS, Probe, R2L, U2R)
    #main("normal", file_name_suffix="mac-normal") # Run the main function with a specific attack type (normal, DoS, Probe, R2L, U2R) and save the results in a specific folder
    # // TODO: 1 mehrere Angreifer-Agenten normal & attack
    # // TODO: 2 Transferability (zuerst nsl-kdd, dann CICIDS2017, dann UNSW-NB15)
    # // TODO: 3. Boruta Tool Document results in Thesis.tex (nutze nur confirmed (ohne automatisch zwang zum entscheiden von tetentative features -> müsste besser abschneiden))
    # // TODO: 4. In R Random Forest Classifier implementieren und testen
    # TODO: lass mal zuerst abwechselnd normal, dann DoS, dann Probe, dann R2L, dann U2R angriffe auswählen für eine bestimmte anzahl an Episoden + Iterationen pro Episode.
    # Idee: zu beginn eine möglichst gleichverteiltes Training durchführen, um den Verteidiger-Agenten zunächst auf alle möglichen Angriffe zu trainieren. 
    # Dadurch soll eine bessere Generalisierung erreicht werden bzw. eine Verzerrung der Daten vermieden werden.
    
    # Idee: 1. 10 Episoden mit normalen angriffen, 10 Episoden mit DoS angriffen, 10 Episoden mit Probe angriffen, 10 Episoden mit R2L angriffen, 10 Episoden mit U2R angriffen
    # anschließend komplett zufällige angriffe auswählen wie bisher.

    # Zweite Idee: belohne den Verteidiger-Agenten stärker, wenn er einen Angriff richtig erkennt im Gegensatz zu einem normalen Zustand. Also korrekter Angriff + 2, korrekter normaler Zustand + 1.
    # -> dadurch wird der Verteidiger-Agent stärker darauf trainiert Angriffe zu erkennen, was in der Praxis auch wichtiger ist.
    # Weiterhin kann ich versuchen Belohnungen dynamisch an die Anzahl der Angriffe anpassen, sodass der Verteidiger-Agent nicht nur auf die Anzahl der Angriffe trainiert wird, sondern auch auf die Art der Angriffe.
    # Messe ob dadurch normale Angriffe gleichbleibend erkannt werden, aber Angriffe (vor allem seltene) besser erkannt werden.
    # Ansatz 1: Statische belohnungen (seltene Angriffe erhalten dennoch eine größere Belohnung)
    # Ansatz 2: Proportionale belohnungen (Anzahl der Angriffe wird berücksichtigt)
        # -> da U2R am wenigsten instanzen hat würde ich die anzahl von iterationen pro episode verkürzen auf 52 (das ist die Anzahl der U2R instanzen)
        # ->
        #

        # blei gleichverteilten Daten scheine ich leicht bessere Ergebnisse zu erhalten (down sampling von normalen instanzen) //TODO: verifizier schaue in meine Ergebnise auf PC (evtl. 2+3 Durchlauf)
        # -> 

    # TODO: Add False Positive Rate for each class. (FP / (FP + TN)) -> FP = False Positives, TN = True Negatives
    # TODO: Prediction Bias für Angriffsklassen berechnen um zu sehen ob der Verteidiger-Agent eine Verzerrung in der Vorhersage hat. Vergleich usprüngliche Verteilung der Angriffe mit der Vorhersage des Verteidiger-Agenten.
    # TODO: Test F
    # TODO: ich könnte anstelle der bisherigen modell ausgaben softmax nutzen, thresholds setzen und damit die vorhersage beeinflussen. -> dadurch könnte ich die FP-Rate für bestimmte Klassen reduzieren.
    # TODO: Data-scheme erstellen. Unit tests schreiben für die Ursprünglichen Daten vs die Formatierten Daten. -> Überprüfen ob die Daten korrekt formatiert wurden. 
        # kategorischen Wert korrekt ? nur eine 1 an der richtigen Stelle ? -> überprüfen ob die Daten korrekt formatiert wurden.
        # Befinden sich alle numerischen Werte im bereich von 0-1 ? -> überprüfen ob die Daten korrekt formatiert wurden.
    # TODO: Schreibe Tests um zu überprüfen ob während des Trainings NaN auftauchen für die weights und layer outputs.
        # Überprüfe auch ob mehr als die Hälfte der Ausgaben einer Schicht != 0 sind.