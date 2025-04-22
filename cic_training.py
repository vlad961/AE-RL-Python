from data.cic_data_manager import CICDataManager
from utils.plot_manager import plot_training_diagrams
from utils.config import CICIDS_2017_CLEAN_ALL_BENIGN, CICIDS_2017_CLEAN_ALL_MALICIOUS, CICIDS_2018_CLEAN_ALL_BENIGN, CICIDS_2018_CLEAN_ALL_MALICIOUS, CWD, TRAINED_MODELS_DIR
from utils.log_config import log_training_parameters, move_log_files, logger_setup, print_end_of_epoch_info_cic, save_debug_info
from utils.helpers import print_total_runtime,  save_trained_models_cic, store_episode_results_cic, store_experience, update_episode_statistics_cic, update_models_and_statistics_cic
from models.rl_env import RLenv
from models.defender_agent import DefenderAgent
from models.attack_agent import AttackAgent
from data.cic_data_manager import cic_attack_map
from datetime import datetime
from test.test_multiple_agents import test_trained_agent_quality_on_inter_set, test_trained_agent_quality_on_intra_set
import gc
import logging
import os, sys
import tensorflow as tf
import time

"""
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
    
    # TODO: 1.2 Trainiere Attack & Defender-Agenten sowohl auf benign als auch auf ein <attack-type> Datensatz.
    #    -> erstelle jeweils Paare für alle Attack Typen. Aggregiere die Resultate und speichere sie in einem Ordner. 
    #   -> Notiere Ergebnisse in der Thesis. Das kommt am aller nähesten An die Experimente aus Towards Generalization in IDS...
    # TODO: 3.1 meine typischen zwei Szenarien. 1. One Attacker: All possible traffic type vs. Defender, capable of classifying every class.; 
    # TODO: 3.2: Multiple Attackers (each benign & attacktype) vs Defender, capable of classifying every class.;
    try:
        batch_size = 1 # Train batch
        minibatch_size = 100 # batch of memory ExpRep
        experience_replay = True
        iterations_episode = 100
        num_episodes = 100
 
        logging.info("Setting up Attacker and Defender Agents...")
        # Load the CICIDS 2017 dataset
        is_cic_2018_trainingset = False
        multiple_attackers = False
        one_vs_all = True
        one_vs_all_target_class = attack_type
        is_inter_dataset_run = True
        data_mgr = CICDataManager(benign_path=CICIDS_2017_CLEAN_ALL_BENIGN, malicious_path=CICIDS_2017_CLEAN_ALL_MALICIOUS, 
                                  is_cic_2018=is_cic_2018_trainingset, one_vs_all=one_vs_all, target_attack_type=one_vs_all_target_class, inter_dataset_run=is_inter_dataset_run, 
                                  inter_dataset_benign_path=CICIDS_2018_CLEAN_ALL_BENIGN, inter_dataset_malicious_path=CICIDS_2018_CLEAN_ALL_MALICIOUS)
        attack_valid_actions = list(range(len(data_mgr.attack_names)))

        # Empirical experience shows that a greater exploration rate is better for the attacker agent.
        att_epsilon = 1
        att_min_epsilon = 0.82  # min value for exploration
        att_gamma = 0.001
        att_decay_rate = 0.99
        att_hidden_size = 100
        att_hidden_layers = 5
        att_learning_rate = 0.001

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
        agent_cic_2017_attacker = AttackAgent(attack_valid_actions, data_mgr.obs_size, "CIC-2017-" + one_vs_all_target_class, "EpsilonGreedy",
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
                                        minibatch_size=100,
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
        log_training_parameters(training_params, attacker_params, defender_params, data_mgr.attack_types, data_mgr.attack_names)
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
            initial_state, _, _ = env.reset()
            # Determine the attack actions for the randomly chosen initial states based on the attackers' policies.
            # Depending on the epsilon value, the attackers either exploit their learned policy to predict the best actions
            # or explore random actions (Exploitation vs. Exploration).
            attack_actions = AttackAgent.get_attack_actions(attackers, initial_state)

            # Retrieve the next states based on the chosen attack actions of the attacker agents.
            # Each state represents the environment after the execution of the corresponding attack.
            # The states are derived from the IDS dataset used in the environment.
            states, _, labels_names = env.get_attack_states(attack_actions)

            # Iteration in one episode
            for iteration in range(iterations_episode):
                attack_indices_list.append(attack_actions)
                attack_names_list.append(labels_names)
                # Determine the defender agent's actions/classifications for the given attack states.
                defender_actions = agent_defender.get_defender_actions(states)

                # Enviroment actuation for those actions
                next_states, _, next_labels_names, def_reward, att_reward, next_attack_actions, done = env.act(defender_actions,
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
        rewards = [def_reward_chain, att_reward_chain]
        losses = [def_loss_chain, att_loss_chain]
        plot_training_diagrams(multiple_attackers, attack_indices_per_episode, cic_attack_map, data_mgr, attacks_mapped_to_att_type_list, plots_path,
            rewards, losses, is_cic_2018_trainingset, agent_defender, attackers, mse_before_history, mae_before_history)

        defender_model_path = os.path.join(TRAINED_MODELS_DIR, f"{output_root_dir}/defender_model.keras")
        test_trained_agent_quality_on_intra_set(defender_model_path, data_mgr, plots_path, one_vs_all=one_vs_all)
        test_trained_agent_quality_on_inter_set(
            path_to_model=defender_model_path,
            x_test=data_mgr.x_val,
            y_test=data_mgr.y_val,
            plots_path=plots_path,
            one_vs_all=True,
            attack_types=data_mgr.attack_types
        )
        move_log_files(current_log_path, destination_log_path)
    except Exception as e:
        logging.error(f"Error occurred\n:{e}", exc_info=True)

# Run the main function
if __name__ == "__main__":
    # CIC-IDS Atack Types: 'Benign', 'Botnet', '(D)DOS', 'Probe', 'Brute Force', 'Web Attack', 'Infiltration', 'Heartbleed'
    main("(D)DOS", file_name_suffix="-Linux-CIC-2017-18-DDOS-1st-Run")