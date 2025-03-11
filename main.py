from models.helpers import download_datasets_if_missing, move_log_files, plot_attack_distributions, plot_rewards_and_losses_during_training, print_total_runtime, save_model, logger_setup
from models.rl_env import RLenv
from models.defender_agent import DefenderAgent
from models.attack_agent import AttackAgent
from data.data_cls import DataCls, attack_types
from datetime import datetime
from test import test_trained_agent_quality


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
    # Store the start time as a datetime object
    script_start_time = datetime.now()

    try:
        download_datasets_if_missing(kdd_train, kdd_test)

        batch_size = 1 # Train batch
        minibatch_size = 100 # batch of memory ExpRep
        ExpRep = True
        iterations_episode = 100
        # num_episodes = int(env.data_shape[0]/(iterations_episode)/10)
        num_episodes = 100

        logging.info("Creating enviroment...")
        if not os.path.exists(formated_train_path) or not os.path.exists(formated_test_path):
            DataCls.format_data(kdd_train, kdd_test, formated_train_path, formated_test_path)

        if(attack_type is not None):
            if attack_type == "equally_balanced_data":
                # Retrieve equally balanced training data (Same amount of Attack and Normal instances).
                data_cls_instance = DataCls(dataset_type='train')
                _, attack_names = data_cls_instance.get_balanced_samples()
            else:
                # Retrieve training data for given attack types and existing attack instances.
                data_cls_instance = DataCls(dataset_type='train')
                _, attack_names = data_cls_instance.get_samples_for_attack_type(attack_type, 0)
        else:
            attack_names = DataCls.get_attack_names(formated_train_path) # Names of attacks in the dataset where at least one sample is present
        
        attack_valid_actions = list(range(len(attack_names)))
        attack_num_actions = len(attack_valid_actions)

        # Empirical experience shows that a greater exploration rate is better for the attacker agent.
        att_epsilon = 1
        att_min_epsilon = 0.82  # min value for exploration
        att_gamma = 0.001
        att_decay_rate = 0.99
        att_hidden_layers = 5
        att_hidden_size = 100
        att_learning_rate = 0.001         #default learning_rate was hardcoded to = 0.00025 on an ADAM optimizer
        obs_size = DataCls.calculate_obs_size(formated_train_path) # Amount of features in the dataset (columns) - attack_types
        

        attacker_agent = AttackAgent(attack_valid_actions, obs_size, "EpsilonGreedy",
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
                                        target_model_name='attacker_target_model',
                                        model_name='attacker_model')

        if attack_type is not None: # If a specific attack is chosen, the training data is loaded with only this attack type
            env = RLenv('train', attacker_agent, batch_size=batch_size, iterations_episode=iterations_episode, specific_attack_type=attack_type, data=data_cls_instance, attack_names=attack_names)
        else:
            env = RLenv('train', attacker_agent, batch_size=batch_size, iterations_episode=iterations_episode)

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

        defender_agent = DefenderAgent(defender_valid_actions, obs_size, "EpsilonGreedy",
                                        epoch_length=iterations_episode,
                                        epsilon=def_epsilon,
                                        min_epsilon=def_min_epsilon,
                                        decay_rate=def_decay_rate,
                                        gamma=def_gamma,
                                        hidden_size=def_hidden_size,
                                        hidden_layers=def_hidden_layers,
                                        minibatch_size=minibatch_size,
                                        mem_size=1000,
                                        learning_rate=def_learning_rate,
                                        ExpRep=ExpRep,
                                        target_model_name='defender_target_model',
                                        model_name='defender_model'
                                        )
        # Statistics
        att_reward_chain = []
        def_reward_chain = []
        att_loss_chain = []
        def_loss_chain = []

        # Print parameters
        logging.info(f"Start Training with the following parameters:\n-----------------------------------------------------------------------------------------------------------------------\n"
                    f"Total epoch: {num_episodes} | Iterations in epoch: {iterations_episode} "
                    f"| Minibatch from mem size: {minibatch_size} | Total Samples: {num_episodes * iterations_episode} | Data shape: {env.data_shape}\n"
                    f"-----------------------------------------------------------------------------------------------------------------------\n"
                    f"Attacker: Num_actions={attack_num_actions} | gamma={att_gamma} | "
                    f"epsilon={att_epsilon} | ANN hidden size={att_hidden_size} | "
                    f"ANN hidden layers={att_hidden_layers} | Learning rate={att_learning_rate}\n"
                    f"-----------------------------------------------------------------------------------------------------------------------\n"
                    f"Defender: Num_actions={defender_num_actions} | gamma={def_gamma} | "
                    f"epsilon={def_epsilon} | ANN hidden size={def_hidden_size} | "
                    f"ANN hidden layers={def_hidden_layers} | Learning rate={def_learning_rate}\n"
                    f"-----------------------------------------------------------------------------------------------------------------------\n"
                    f"Used Attack types: '{attack_type if attack_type is not None else 'all: Normal, DoS, Probe, R2L, U2R'}' | Attack name(s): {attack_names}\n"
                    f"-----------------------------------------------------------------------------------------------------------------------\n"
                    )
        
        # Main loops
        attacks_by_epoch = []
        attack_labels_list = []
        for epoch in range(num_episodes):
            epoch_start_time = time.time()
            att_loss = 0.0
            def_loss = 0.0
            def_total_reward_by_episode = 0
            att_total_reward_by_episode = 0
            # Reset enviromet, actualize the data batch with random states/attacks.
            states = env.reset()
            # Get action(s)/attack(s) for randomly chosen state(s)/attack(s) following the policy of the attacker.
            # The attacker's policy can either predict the best possible action(s)/attack(s) with respect to the randomly chosen state(s)/attack(s)
            # or explore random action(s)/attack(s) based on the epsilon value (Exploitation vs. Exploration).
            attack_actions = attacker_agent.act(states)
            # Based on the chosen actions/attacks, the next states are determined.
            # These states represent the environment after the attacker agent's actions/attacks have been executed.
            states = env.get_states(attack_actions)

            done = False
# // TODO: 1 mehrere Angreifer-Agenten normal & attack
# // TODO: 2 Transferability (zuerst nsl-kdd, dann CICIDS2017, dann UNSW-NB15)
# // TODO: 3. Boruta Tool Document results in Thesis.tex (nutze nur confirmed (ohne automatisch zwang zum entscheiden von tetentative features -> müsste besser abschneiden))
# // TODO: 4. In R Random Forest Classifier implementieren und testen
            attacks_list = []
            # Iteration in one episode
            for i_iteration in range(iterations_episode):
                attacks_list.append(attack_actions[0])
                # Get action(s)/classification(s) of the defender agent for the chosen state(s)/attack(s).
                defender_actions = defender_agent.act(states)
                # Enviroment actuation for this actions
                next_states, def_reward, att_reward, next_attack_actions, done = env.act(defender_actions,
                                                                                            attack_actions)

                attacker_agent.learn(states, attack_actions, next_states, att_reward, done)
                defender_agent.learn(states, defender_actions, next_states, def_reward, done)

                # Train network, update loss after at least minibatch_learns
                if ExpRep and epoch * iterations_episode + i_iteration >= minibatch_size:
                    def_loss += defender_agent.update_model()[0]
                    att_loss += attacker_agent.update_model()[0]
                elif not ExpRep:
                    def_loss += defender_agent.update_model()[0]
                    att_loss += attacker_agent.update_model()[0]

                # Update the state
                states = next_states
                attack_actions = next_attack_actions

                # Update statistics
                def_total_reward_by_episode += np.sum(def_reward, dtype=np.int32)
                att_total_reward_by_episode += np.sum(att_reward, dtype=np.int32)

            attacks_by_epoch.append(attacks_list)
            # Update user view
            def_reward_chain.append(def_total_reward_by_episode)
            att_reward_chain.append(att_total_reward_by_episode)
            def_loss_chain.append(def_loss)
            att_loss_chain.append(att_loss)

            end_time = time.time()
            logging.info(f"End of Epoch.\r\n|Epoch {epoch:03d}/{num_episodes:03d}| time: {(end_time - epoch_start_time):2.2f}|\r\n"
                    f"|Def Loss {def_loss:4.4f} | Def Reward in ep {def_total_reward_by_episode:03d}|\r\n"
                    f"|Att Loss {att_loss:4.4f} | Att Reward in ep {att_total_reward_by_episode:03d}|\r\n"
                    f"|Def Estimated: {env.def_estimated_labels}| Att Labels: {env.att_true_labels}|\r\n"
                    f"|Def Amount of true predicted attacks: {env.def_true_labels}|")
            attack_labels_list.append(env.att_true_labels)

        # Save the trained models
        attacker_model_path = os.path.join(trained_models_dir, f"{output_root_dir}/attacker_model.keras")
        defender_model_path = os.path.join(trained_models_dir, f"{output_root_dir}/defender_model.keras")

        save_model(attacker_agent, attacker_model_path)
        save_model(defender_agent, defender_model_path)
        logging.info("Saved trained models.")
        print_total_runtime(script_start_time)

        # Test and visualize results
        plots_path = os.path.join(trained_models_dir, f"{output_root_dir}/plots/")
        current_log_path = os.path.join(cwd, f"logs/{output_root_dir}.log")
        destination_log_path = os.path.join(trained_models_dir, f"{output_root_dir}/logs/")
        logging.info(f"Trying to save summary plots under: {plots_path}")
        plot_rewards_and_losses_during_training(def_reward_chain, att_reward_chain, def_loss_chain, att_loss_chain, plots_path)
        plot_attack_distributions(attacks_by_epoch, env.attack_names, attack_labels_list, plots_path)
        test_trained_agent_quality(defender_model_path, plots_path)
        move_log_files(current_log_path, destination_log_path)
    except Exception as e:
        logging.error(f"Error occurred\n:{e}", exc_info=True)

# Run the main function
if __name__ == "__main__":
    main(file_name_suffix="-WIN-balanced-data-att-5L-def-3L-lr-0.001")
    #main("U2R", file_name_suffix="-WIN-only-DoS")
    #main("equally_balanced_data", file_name_suffix="-WIN-equally-balanced-data")
    #main(["normal", "R2L", "U2R"], file_name_suffix="-WIN-normal-r2l-u2r-attacks-att-5L-def-3L-lr-0.001") # Run the main function with a list of specific attack types (normal, DoS, Probe, R2L, U2R)
    #main(["normal", "U2R"], file_name_suffix="-WIN-normal-U2R") # Run the main function with a list of specific attack types (normal, DoS, Probe, R2L, U2R)
    #main() # Run the main function with all attack types (normal, DoS, Probe, R2L, U2R) and save the results in a default folder (timestamp).
    #main(file_name_suffix="mac-all-attacks") # Run the main function with all attack types and save the results in a specific folder (mac-all-attacks)
    #main("normal") # Run the main function with a specific attack type (normal, DoS, Probe, R2L, U2R)
    #main("normal", file_name_suffix="mac-normal") # Run the main function with a specific attack type (normal, DoS, Probe, R2L, U2R) and save the results in a specific folder
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