
from typing import List
from log_config import log_training_parameters, print_end_of_epoch_info, move_log_files, logger_setup
from models.helpers import create_attack_id_to_index_mapping, create_attack_id_to_type_mapping, get_attack_actions, get_attack_states, get_defender_actions, getAttackTypeMaps, print_total_runtime, save_trained_models, store_episode_results, store_experience, transform_attacks_by_epoch, transform_attacks_by_type, update_episode_statistics, update_models_and_statistics
from models.rl_env import RLenv
from models.defender_agent import DefenderAgent
from models.attack_agent import AttackAgent
from data.data_cls import DataCls, attack_types, attack_map
from datetime import datetime
from test_multiple_agents import test_trained_agent_quality
from plotting_multiple_agents import plot_attack_distribution_for_each_attacker, plot_attack_distributions_multiple_agents, plot_mapped_attack_distribution_for_each_attacker, plot_rewards_and_losses_during_training_multiple_agents, plot_rewards_losses_boxplot, plot_training_error, plot_trend_lines_multiple_agents
import logging
import numpy as np
import time
import os

"""
This script is the main entry point for the project. It is responsible for downloading the data, training the agents and saving the trained models.

Notes and credits:
The following anomaly detection RL system is based on the work of Guillermo Caminero, Manuel Lopez-Martin, Belen Carro "Adversarial environment reinforcement learning algorithm for intrusion detection".
The original project can be found at: https://github.com/gcamfer/Anomaly-ReactionRL
To be more specific, the code is based on the following file: 'NSL-KDD adaption: AE_RL_NSL-KDD.ipynb' https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/Notebooks/AE_RL_NSL_KDD.ipynb
"""

cwd = os.getcwd()
data_root_dir = os.path.join(cwd, "data/datasets/")
data_original_dir = os.path.join(data_root_dir, "origin-kaggle-com/nsl-kdd/")
data_formated_dir = os.path.join(data_root_dir, "formated/")
formated_train_path = os.path.join(data_formated_dir, "balanced_training_data.csv") # formated_train_adv
formated_test_path = os.path.join(data_formated_dir, "balanced_test_data.csv") # formated_test_adv
kdd_train = os.path.join(data_original_dir, "KDDTrain+.txt")
kdd_test = os.path.join(data_original_dir, "KDDTest+.txt")
trained_models_dir = os.path.join(cwd, "models/trained-models/")

def main(attack_type=None, file_name_suffix=""):
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_root_dir = f"{timestamp_begin}{file_name_suffix}"
    logger_setup(timestamp_begin, file_name_suffix)
    logging.info(f"Started script at: {timestamp_begin}")
    logging.info(f"Current working dir: {cwd}")
    logging.info(f"Used data files: \n{kdd_train},\n{kdd_test}")
    script_start_time = datetime.now()

    try:
        batch_size = 1 # Train batch
        minibatch_size = 100 # batch of memory ExpRep
        ExpRep = True
        iterations_episode = 100
        num_episodes = 50 # //FIXME: 100 after validate correct running

        logging.info("Creating enviroment...")
        if not os.path.exists(formated_train_path) or not os.path.exists(formated_test_path):
            DataCls.format_data("train", kdd_train, kdd_test, formated_train_path, formated_test_path)

        attack_names = DataCls.get_attack_names(formated_train_path) # Names of attacks in the dataset where at least one sample is present
        attack_valid_actions = list(range(len(attack_names)))
        attack_num_actions = len(attack_valid_actions)

        attack_valid_actions_dos, attack_valid_actions_probe, attack_valid_actions_r2l, attack_valid_actions_u2r = getAttackTypeMaps(attack_map, attack_names)

        # Empirical experience shows that a greater exploration rate is better for the attacker agent.
        att_epsilon = 1
        att_min_epsilon = 0.82  # min value for exploration
        att_gamma = 0.001
        att_decay_rate = 0.99
        att_hidden_layers = 5
        att_hidden_size = 100
        att_learning_rate = 0.001
        obs_size = DataCls.calculate_obs_size(formated_train_path) # Amount of features in the dataset (columns) - attack_types

        agent_dos = AttackAgent(attack_valid_actions_dos, obs_size, "DoS", "EpsilonGreedy",
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
                                        ExpRep=ExpRep,
                                        target_model_name='attacker_target_model_dos',
                                        model_name='attacker_model_dos')
        
        agent_probe = AttackAgent(attack_valid_actions_probe, obs_size, "Probe", "EpsilonGreedy",
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
                                ExpRep=ExpRep,
                                target_model_name='attacker_target_model_probe',
                                model_name='attacker_model_probe')
        
        agent_r2l = AttackAgent(attack_valid_actions_r2l, obs_size, "R2L", "EpsilonGreedy",
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
                ExpRep=ExpRep,
                target_model_name='attacker_target_model_r2l',
                model_name='attacker_model_probe_r2l')
        
        agent_u2r = AttackAgent(attack_valid_actions_u2r, obs_size, "U2R", "EpsilonGreedy",
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
                        ExpRep=ExpRep,
                        target_model_name='attacker_target_model_u2r',
                        model_name='attacker_model_probe_u2r')
        
        attackers = [agent_dos, agent_probe, agent_r2l, agent_u2r]

        env = RLenv('train', attackers, batch_size=batch_size, iterations_episode=iterations_episode)

        # Defender is trained to detect type of attacks 0: normal, 1: Dos, 2: Probe, 3: R2L, 4: U2R
        defender_valid_actions = list(range(len(attack_types)))
        defender_num_actions = len(defender_valid_actions)
        def_epsilon = 1  # exploration
        def_min_epsilon = 0.01  # min value for exploration
        def_gamma = 0.001
        def_decay_rate = 0.99
        def_hidden_size = 100
        def_hidden_layers = 3
        def_learning_rate = 0.001

        agent_defender = DefenderAgent(defender_valid_actions, obs_size, "EpsilonGreedy",
                                        epoch_length=iterations_episode,
                                        epsilon=def_epsilon,
                                        min_epsilon=def_min_epsilon,
                                        decay_rate=def_decay_rate,
                                        gamma=def_gamma,
                                        hidden_size=def_hidden_size,
                                        hidden_layers=def_hidden_layers,
                                        minibatch_size=400, # //TODO: auf 400 setzen, da 4 agenten?
                                        mem_size=4000,
                                        learning_rate=def_learning_rate,
                                        ExpRep=ExpRep,
                                        target_model_name='defender_target_model',
                                        model_name='defender_model'
                                        )
        
        # Print training parameters
        log_training_parameters(num_episodes, iterations_episode, minibatch_size, env, attack_num_actions,
                                att_gamma, att_epsilon, att_hidden_size, att_hidden_layers, att_learning_rate,
                                defender_num_actions, def_gamma, def_epsilon, def_hidden_size, def_hidden_layers,
                                def_learning_rate, attack_type, attack_names)
        # Statistics
        att_reward_chain, att_reward_chain_dos, att_reward_chain_probe, att_reward_chain_r2l, att_reward_chain_u2r = [], [], [], [], []
        att_loss_chain, att_loss_chain_dos, att_loss_chain_probe, att_loss_chain_r2l, att_loss_chain_u2r = [], [], [], [], []
        def_reward_chain = []
        def_loss_chain = []
        def_metrics_chain, att_metrics_chain = [], []
        mse_before_history, mae_before_history = [], []
        mse_after_history, mae_after_history = [], []
        
        # Main loops
        attack_indices_per_episode, attack_names_per_episode = [], []
        attacks_mapped_to_att_type_list = []

        for episode in range(num_episodes):
            epoch_start_time = time.time()
            epoch_mse_before, epoch_mae_before = [], []
            epoch_mse_after, epoch_mae_after = [], []
            # Attack and defense losses
            att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r = 0.0, 0.0, 0.0, 0.0
            def_loss, agg_att_loss = 0.0, 0.0
            # Total rewards by episode / Total rewards by episode for each attack type
            def_total_reward_by_episode, att_total_reward_by_episode = 0, 0
            att_total_reward_by_episode_dos, att_total_reward_by_episode_probe, att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r = 0, 0, 0, 0
            
            attack_indices_list = []
            attack_names_list = []
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

                # Train network, update loss after at least minibatch_size learns (observations)
                if ExpRep and episode * iterations_episode + iteration >= minibatch_size:
                    def_loss, att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, agg_att_loss = update_models_and_statistics(
                            agent_defender, attackers, def_loss, att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, agg_att_loss, def_metrics_chain, 
                            att_metrics_chain, epoch_mse_before, epoch_mae_before, epoch_mse_after, epoch_mae_after)

                # Update the environment for the next iteration
                states = next_states
                attack_actions = next_attack_actions
                # Update the labels for the next iteration used only for debugging reasons for a better understanding of the agent's behavior
                labels_names = next_labels_names

                # Update episode statistics
                (def_total_reward_by_episode, att_total_reward_by_episode, att_total_reward_by_episode_dos, att_total_reward_by_episode_probe, 
                    att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r) = update_episode_statistics(def_reward, att_reward, 
                            def_total_reward_by_episode, att_total_reward_by_episode, att_total_reward_by_episode_dos, att_total_reward_by_episode_probe,
                            att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r)

            # Store episode results
            store_episode_results(attack_indices_list, attack_names_list, env, epoch_mse_before, epoch_mae_before,
                                epoch_mse_after, epoch_mae_after, def_total_reward_by_episode, att_total_reward_by_episode,
                                att_total_reward_by_episode_dos, att_total_reward_by_episode_probe,
                                att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r, def_loss, agg_att_loss,
                                att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, attack_indices_per_episode,
                                attack_names_per_episode, attacks_mapped_to_att_type_list, mse_before_history,
                                mae_before_history, mse_after_history, mae_after_history, def_reward_chain,
                                att_reward_chain, att_reward_chain_dos, att_reward_chain_probe, att_reward_chain_r2l,
                                att_reward_chain_u2r, def_loss_chain, att_loss_chain, att_loss_chain_dos,
                                att_loss_chain_probe, att_loss_chain_r2l, att_loss_chain_u2r)

            end_time = time.time()
            # Update user view
            print_end_of_epoch_info(episode, num_episodes, epoch_start_time, end_time, def_loss, def_total_reward_by_episode, att_loss_dos, att_total_reward_by_episode_dos, att_loss_probe, att_total_reward_by_episode_probe, att_loss_r2l, att_total_reward_by_episode_r2l, att_loss_u2r, att_total_reward_by_episode_u2r, env)

        # Save the trained models
        agents = {
            "dos": agent_dos,
            "probe": agent_probe,
            "r2l": agent_r2l,
            "u2r": agent_u2r,
            "defender": agent_defender,
        }
        save_trained_models(agents, output_root_dir, trained_models_dir)        
        print_total_runtime(script_start_time)

        # Test and visualize results
        plots_path = os.path.join(trained_models_dir, f"{output_root_dir}/plots/")
        current_log_path = os.path.join(cwd, f"logs/{output_root_dir}.log")
        destination_log_path = os.path.join(trained_models_dir, f"{output_root_dir}/logs/")
        logging.info(f"Trying to save summary plots under: {plots_path}")
        plot_rewards_and_losses_during_training_multiple_agents(def_reward_chain, att_reward_chain, def_loss_chain, att_loss_chain, plots_path)
        plot_attack_distributions_multiple_agents(attack_indices_per_episode, attack_map, env.attack_names, attacks_mapped_to_att_type_list, plots_path)
        rewards = [def_reward_chain, att_reward_chain_dos, att_reward_chain_probe, att_reward_chain_r2l, att_reward_chain_u2r]
        losses = [def_loss_chain, att_loss_chain_dos, att_loss_chain_probe, att_loss_chain_r2l, att_loss_chain_u2r]
        plot_trend_lines_multiple_agents(rewards, losses, [agent_defender.name, agent_dos.name, agent_probe.name, agent_r2l.name, agent_u2r.name], plots_path)

        attack_id_to_index = create_attack_id_to_index_mapping(attack_map, attack_names)
        num_attacks = len(attack_names)
        transformed_attacks = transform_attacks_by_epoch(attack_indices_per_episode, attack_id_to_index, num_attacks)
        plot_attack_distribution_for_each_attacker(transformed_attacks, attack_names, plots_path, ['Attacker DoS', 'Attacker Probe', 'Attacker R2L', 'Attacker U2R'])
        
        attack_id_to_type = create_attack_id_to_type_mapping(attack_map)
        # Daten transformieren
        transformed_attacks_by_type = transform_attacks_by_type(attack_indices_per_episode, attack_id_to_type, attack_types)

        plot_mapped_attack_distribution_for_each_attacker(transformed_attacks_by_type, ['Normal','Dos','Probe','R2L', 'U2R'], plots_path, ['Attacker DoS', 'Attacker Probe', 'Attacker R2L', 'Attacker U2R'])
        plot_rewards_losses_boxplot(rewards, losses, [agent_defender.name, agent_dos.name, agent_probe.name, agent_r2l.name, agent_u2r.name], plots_path)# FIXME: überprüfe ob Logik korrekt ist. 
        # Nutze die Funktion nach Abschluss deines Trainings:
        plot_training_error(mse_before_history, mae_before_history, mse_after_history, mae_after_history, save_path=plots_path)
        
        defender_model_path = os.path.join(trained_models_dir, f"{output_root_dir}/defender_model.keras")
        test_trained_agent_quality(defender_model_path, plots_path)
        move_log_files(current_log_path, destination_log_path)
    except Exception as e:
        logging.error(f"Error occurred\n:{e}", exc_info=True)

# Run the main function
if __name__ == "__main__":
    main(file_name_suffix="-WIN-multiple-attackers-balanced-data-att-5L-def-3L-lr-0.001")
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