from utils.config import CWD, NSL_KDD_FORMATTED_TEST_PATH, NSL_KDD_FORMATTED_TRAIN_PATH, ORIGINAL_KDD_TEST, ORIGINAL_KDD_TRAIN, TRAINED_MODELS_DIR
from utils.log_config import log_training_parameters, print_end_of_epoch_info, move_log_files, logger_setup, save_debug_info
from utils.helpers import create_attack_id_to_index_mapping, create_attack_id_to_type_mapping, get_attack_type_maps, print_total_runtime, save_trained_models, store_episode_results, store_experience, transform_attacks_by_epoch, transform_attacks_by_type, update_episode_statistics, update_models_and_statistics
from models.rl_env import RLenv
from models.defender_agent import DefenderAgent
from models.attack_agent import AttackAgent
from data.nsl_kdd_data_manager import NslKddDataManager, attack_types, nsl_kdd_attack_map
from datetime import datetime
from test.test_multiple_agents import test_trained_agent_quality_on_intra_set
from utils.plotting_multiple_agents import plot_attack_distribution_for_each_attacker, plot_attack_distributions_multiple_agents, plot_mapped_attack_distribution_for_each_attacker, plot_rewards_and_losses_during_training_multiple_agents, plot_rewards_losses_boxplot, plot_training_error, plot_trend_lines_multiple_agents
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
    logging.info(f"Started script at: {timestamp_begin}\nCurrent working dir: {CWD}\nUsed data files: \n{ORIGINAL_KDD_TRAIN},\n{ORIGINAL_KDD_TEST}")
    script_start_time = datetime.now()

    try:
        batch_size = 1 # Train batch
        minibatch_size_attacker = 100 # batch of memory ExpRep
        minibatch_size_defender = 400 # batch of memory ExpRep
        mem_size_attacker = 1000 # total memory size ExpRep before overwriting first experiences
        mem_size_defender = 4000 # total memory size ExpRep before overwriting first experiences
        experience_replay = True
        iterations_episode = 100
        multiple_attackers = True
        num_episodes = 100
 
        logging.info("Setting up Attacker and Defender Agents...")

        train_data = NslKddDataManager(ORIGINAL_KDD_TRAIN, ORIGINAL_KDD_TEST, NSL_KDD_FORMATTED_TRAIN_PATH, NSL_KDD_FORMATTED_TEST_PATH, normalization='linear', multiple_attackers=multiple_attackers)
        attack_valid_actions = list(range(len(train_data.attack_names)))
        attack_valid_actions_dos, attack_valid_actions_probe, attack_valid_actions_r2l, attack_valid_actions_u2r = get_attack_type_maps(nsl_kdd_attack_map, train_data.attack_names)

        # Empirical experience shows that a greater exploration rate is better for the attacker agent.
        att_epsilon = 1
        att_min_epsilon = 0.82  # min value for exploration
        att_gamma = 0.001
        att_decay_rate = 0.99
        att_hidden_size = 100
        att_hidden_layers = 5
        att_learning_rate = 0.001

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

        training_params = {"num_episodes": num_episodes, "iterations_episode": iterations_episode, "minibatch_size_attacker": minibatch_size_attacker,
                           "minibatch_size_defender": minibatch_size_defender,
                           "total_samples": num_episodes * iterations_episode, "data_shape": train_data.shape}
        attacker_params = {"num_actions": len(attack_valid_actions),"gamma": att_gamma, "epsilon": att_epsilon, "hidden_size": att_hidden_size,
                           "hidden_layers": att_hidden_layers, "learning_rate": att_learning_rate }
        defender_params = {"num_actions": defender_num_actions, "gamma": def_gamma, "epsilon": def_epsilon, "hidden_size": def_hidden_size,
                           "hidden_layers": def_hidden_layers,"learning_rate": def_learning_rate}

        # Create the attacker and defender agents
        agent_dos = AttackAgent(attack_valid_actions_dos, train_data.obs_size, "DoS", "EpsilonGreedy",
                                        epoch_length=iterations_episode,
                                        epsilon=att_epsilon,
                                        min_epsilon=att_min_epsilon,
                                        decay_rate=att_decay_rate,
                                        gamma=att_gamma,
                                        hidden_size=att_hidden_size,
                                        hidden_layers=att_hidden_layers,
                                        minibatch_size=minibatch_size_attacker,
                                        mem_size=mem_size_attacker,
                                        learning_rate=att_learning_rate,
                                        ExpRep=experience_replay,
                                        target_model_name='attacker_target_model_dos',
                                        model_name='attacker_model_dos')
        
        agent_probe = AttackAgent(attack_valid_actions_probe, train_data.obs_size, "Probe", "EpsilonGreedy",
                                epoch_length=iterations_episode,
                                epsilon=att_epsilon,
                                min_epsilon=att_min_epsilon,
                                decay_rate=att_decay_rate,
                                gamma=att_gamma,
                                hidden_size=att_hidden_size,
                                hidden_layers=att_hidden_layers,
                                minibatch_size=minibatch_size_attacker,
                                mem_size=mem_size_attacker,
                                learning_rate=att_learning_rate,
                                ExpRep=experience_replay,
                                target_model_name='attacker_target_model_probe',
                                model_name='attacker_model_probe')
        
        agent_r2l = AttackAgent(attack_valid_actions_r2l, train_data.obs_size, "R2L", "EpsilonGreedy",
                epoch_length=iterations_episode,
                epsilon=att_epsilon,
                min_epsilon=att_min_epsilon,
                decay_rate=att_decay_rate,
                gamma=att_gamma,
                hidden_size=att_hidden_size,
                hidden_layers=att_hidden_layers,
                minibatch_size=minibatch_size_attacker,
                mem_size=mem_size_attacker,
                learning_rate=att_learning_rate,
                ExpRep=experience_replay,
                target_model_name='attacker_target_model_r2l',
                model_name='attacker_model_probe_r2l')
        
        agent_u2r = AttackAgent(attack_valid_actions_u2r, train_data.obs_size, "U2R", "EpsilonGreedy",
                        epoch_length=iterations_episode,
                        epsilon=att_epsilon,
                        min_epsilon=att_min_epsilon,
                        decay_rate=att_decay_rate,
                        gamma=att_gamma,
                        hidden_size=att_hidden_size,
                        hidden_layers=att_hidden_layers,
                        minibatch_size=minibatch_size_attacker,
                        mem_size=mem_size_attacker,
                        learning_rate=att_learning_rate,
                        ExpRep=experience_replay,
                        target_model_name='attacker_target_model_u2r',
                        model_name='attacker_model_probe_u2r')
        
        attackers = [agent_dos, agent_probe, agent_r2l, agent_u2r]

        agent_defender = DefenderAgent(defender_valid_actions, train_data.obs_size, "EpsilonGreedy",
                                        epoch_length=iterations_episode,
                                        epsilon=def_epsilon,
                                        min_epsilon=def_min_epsilon,
                                        decay_rate=def_decay_rate,
                                        gamma=def_gamma,
                                        hidden_size=def_hidden_size,
                                        hidden_layers=def_hidden_layers,
                                        minibatch_size=minibatch_size_defender, # //TODO: auf 400 setzen, da 4 agenten?
                                        mem_size=mem_size_defender,
                                        learning_rate=def_learning_rate,
                                        ExpRep=experience_replay,
                                        target_model_name='defender_target_model',
                                        model_name='defender_model'
                                        )

        # Create the environment
        logging.info("Creating environment...")
        env = RLenv(train_data, attackers, agent_defender, batch_size=batch_size, iterations_episode=iterations_episode)

        # Print training parameters
        log_training_parameters(training_params, attacker_params, defender_params, attack_type, train_data.attack_names)
        # Statistics
        att_reward_chain, att_reward_chain_dos, att_reward_chain_probe, att_reward_chain_r2l, att_reward_chain_u2r = [], [], [], [], []
        att_loss_chain, att_loss_chain_dos, att_loss_chain_probe, att_loss_chain_r2l, att_loss_chain_u2r = [], [], [], [], []
        def_reward_chain = []
        def_loss_chain = []
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
            att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r = 0.0, 0.0, 0.0, 0.0
            def_loss, agg_att_loss = 0.0, 0.0
            # Total rewards by episode / Total rewards by episode for each attack type
            def_total_reward_by_episode, att_total_reward_by_episode = 0, 0
            att_total_reward_by_episode_dos, att_total_reward_by_episode_probe, att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r = 0, 0, 0, 0
            
            attack_indices_list = []
            attack_names_list = []
            sample_indices_list = []
            done = False
            
            # Reset the environment and initialize a new batch of random states and corresponding attack labels.
            initial_state, _ = env.reset()
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
                if experience_replay and episode * iterations_episode + iteration >= minibatch_size_attacker:
                    
                    def_loss, att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, agg_att_loss = update_models_and_statistics(
                            agent_defender, attackers, def_loss, att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, agg_att_loss, def_metrics_chain, 
                            att_metrics_chain, epoch_mse_before, epoch_mae_before, sample_indices_list)

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
                                def_total_reward_by_episode, att_total_reward_by_episode,
                                att_total_reward_by_episode_dos, att_total_reward_by_episode_probe,
                                att_total_reward_by_episode_r2l, att_total_reward_by_episode_u2r, def_loss, agg_att_loss,
                                att_loss_dos, att_loss_probe, att_loss_r2l, att_loss_u2r, attack_indices_per_episode,
                                attack_names_per_episode, attacks_mapped_to_att_type_list, mse_before_history,
                                mae_before_history, def_reward_chain,
                                att_reward_chain, att_reward_chain_dos, att_reward_chain_probe, att_reward_chain_r2l,
                                att_reward_chain_u2r, def_loss_chain, att_loss_chain, att_loss_chain_dos,
                                att_loss_chain_probe, att_loss_chain_r2l, att_loss_chain_u2r, sample_indices_per_episode, sample_indices_list)

            end_time = time.time()
            # Update user view
            episode_info = {"episode": episode, "num_episodes": num_episodes, "epoch_start_time": epoch_start_time, "end_time": end_time}
            metrics = {"def_loss": def_loss, "def_total_reward_by_episode": def_total_reward_by_episode,
                       "att_loss_dos": att_loss_dos, "att_total_reward_by_episode_dos": att_total_reward_by_episode_dos,
                       "att_loss_probe": att_loss_probe, "att_total_reward_by_episode_probe": att_total_reward_by_episode_probe,
                       "att_loss_r2l": att_loss_r2l, "att_total_reward_by_episode_r2l": att_total_reward_by_episode_r2l,
                       "att_loss_u2r": att_loss_u2r, "att_total_reward_by_episode_u2r": att_total_reward_by_episode_u2r}

            print_end_of_epoch_info(episode_info, metrics, env)
            gc.collect()

        # Save the trained models
        agents = {"dos": agent_dos, "probe": agent_probe, "r2l": agent_r2l,"u2r": agent_u2r, "defender": agent_defender}
        save_trained_models(agents, output_root_dir, TRAINED_MODELS_DIR)
        print_total_runtime(script_start_time)

        #################################
        # Collect all relevant variables#
        #################################
        debug_info = {
            "rewards": {
                "defender": def_reward_chain,
                "aggregated_attacker": att_reward_chain,
                "dos": att_reward_chain_dos,
                "probe": att_reward_chain_probe,
                "r2l": att_reward_chain_r2l,
                "u2r": att_reward_chain_u2r,
            },
            "losses": {
                "defender": def_loss_chain,
                "aggregated_attacker": att_loss_chain,
                "dos": att_loss_chain_dos,
                "probe": att_loss_chain_probe,
                "r2l": att_loss_chain_r2l,
                "u2r": att_loss_chain_u2r,
            },
            "attack_indices_per_episode": attack_indices_per_episode,
            "attack_names_per_episode": attack_names_per_episode,
            "attacks_mapped_to_att_type_list": attacks_mapped_to_att_type_list,
            "mse_before_history": mse_before_history,
            "mae_before_history": mae_before_history,
            "agents": {
                "defender": agent_defender.name,
                "dos": agent_dos.name,
                "probe": agent_probe.name,
                "r2l": agent_r2l.name,
                "u2r": agent_u2r.name,
            },
            "sample_indices_per_episode": sample_indices_per_episode,
            "attack_id_to_index": create_attack_id_to_index_mapping(nsl_kdd_attack_map, train_data.attack_names),
            "attack_id_to_type": create_attack_id_to_type_mapping(nsl_kdd_attack_map),
            "attack_names": train_data.attack_names,
            "attack_types": attack_types,
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
        plot_attack_distributions_multiple_agents(attack_indices_per_episode, nsl_kdd_attack_map, train_data.attack_names, attacks_mapped_to_att_type_list, plots_path)
        rewards = [def_reward_chain, att_reward_chain_dos, att_reward_chain_probe, att_reward_chain_r2l, att_reward_chain_u2r]
        losses = [def_loss_chain, att_loss_chain_dos, att_loss_chain_probe, att_loss_chain_r2l, att_loss_chain_u2r]
        plot_trend_lines_multiple_agents(rewards, losses, [agent_defender.name, agent_dos.name, agent_probe.name, agent_r2l.name, agent_u2r.name], plots_path)

        attack_id_to_index = create_attack_id_to_index_mapping(nsl_kdd_attack_map, train_data.attack_names)
        num_attacks = len(train_data.attack_names)
        transformed_attacks = transform_attacks_by_epoch(attack_indices_per_episode, attack_id_to_index, num_attacks)
        plot_attack_distribution_for_each_attacker(transformed_attacks, train_data.attack_names, plots_path, [attacker.name for attacker in attackers])
        
        attack_id_to_type = create_attack_id_to_type_mapping(nsl_kdd_attack_map)
        # Daten transformieren
        transformed_attacks_by_type = transform_attacks_by_type(attack_indices_per_episode, attack_id_to_type, attack_types)

        plot_mapped_attack_distribution_for_each_attacker(transformed_attacks_by_type, train_data.attack_types, plots_path, [attacker.name for attacker in attackers])
        plot_rewards_losses_boxplot(rewards, losses, [agent_defender.name, agent_dos.name, agent_probe.name, agent_r2l.name, agent_u2r.name], plots_path)
        # Nutze die Funktion nach Abschluss deines Trainings:
        plot_training_error(mse_before_history, mae_before_history, save_path=plots_path)
        
        defender_model_path = os.path.join(TRAINED_MODELS_DIR, f"{output_root_dir}/defender_model.keras")
        test_trained_agent_quality_on_intra_set(defender_model_path, train_data, plots_path)
        move_log_files(current_log_path, destination_log_path)
    except Exception as e:
        logging.error(f"Error occurred\n:{e}", exc_info=True)

# Run the main function
if __name__ == "__main__":
    main(file_name_suffix="-multi-attacker-nsl-def-mem-4000-minibatch-400-3rd")