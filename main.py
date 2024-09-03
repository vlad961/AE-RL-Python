from models.rl_env import RLenv
from models.defender_agent import DefenderAgent
from models.attack_agent import AttackAgent
from models.q_network import CustomHuberLoss
from data.data_cls import DataCls

import numpy as np
import time
import os
import requests

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def save_model(agent, model_path):
    agent.model_network.model.save(model_path)

def load_model(agent, model_path):
    agent.model_network.model.load_model(model_path, custom_objects={'CustomHuberLoss': CustomHuberLoss})

if __name__ == "__main__":
    print("Current working dir:", os.getcwd())
    kdd_train_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTrain%2B.txt"
    kdd_test_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTest%2B.txt"

    # Download the dataset
    data_dir = os.path.join(os.getcwd(), "data/datasets/NSL")
    kdd_train = os.path.join(data_dir, "KDDTrain.txt")
    kdd_test = os.path.join(data_dir, "KDDTest.txt")

    attacker_model_path = os.path.join(os.getcwd(), "models/attacker_model.h5")
    defender_model_path = os.path.join(os.getcwd(), "models/defender_model.h5")

    download_file(kdd_train_url, kdd_train)
    download_file(kdd_test_url, kdd_test)
    print("Downloaded the dataset\n")

    formated_train_path = os.path.join(data_dir, "formated_train_adv.data")
    formated_test_path = os.path.join(data_dir, "formated_test_adv.data")

    # Train batch
    batch_size = 1
    # batch of memory ExpRep
    minibatch_size = 100
    ExpRep = True

    iterations_episode = 100

    # Initialization of the enviroment
    print("Creating enviroment...")

    # Initialize the AttackAgent
    '''
    Definition for the attacker agent.
    In this case the exploration is better to be greater
    The correlation sould be greater too so gamma bigger
    '''
    attack_names = DataCls.get_attack_names(formated_train_path)
    attack_valid_actions = list(range(len(attack_names)))
    attack_num_actions = len(attack_valid_actions)

    att_epsilon = 1
    min_epsilon = 0.82  # min value for exploration

    att_gamma = 0.001
    att_decay_rate = 0.99

    att_hidden_layers = 1
    att_hidden_size = 100

    att_learning_rate = 0.2
    obs_size = DataCls.calculate_obs_size(formated_train_path)
    

    attacker_agent = AttackAgent(attack_valid_actions, obs_size, "EpsilonGreedy",
                                    epoch_length=iterations_episode,
                                    epsilon=att_epsilon,
                                    min_epsilon=min_epsilon,
                                    decay_rate=att_decay_rate,
                                    gamma=att_gamma,
                                    hidden_size=att_hidden_size,
                                    hidden_layers=att_hidden_layers,
                                    minibatch_size=minibatch_size,
                                    mem_size=1000,
                                    learning_rate=att_learning_rate,
                                    ExpRep=ExpRep)


    env = RLenv('train', train_path=kdd_train, attack_agent=attacker_agent, test_path=kdd_test,
                formated_train_path=formated_train_path,
                formated_test_path=formated_test_path, batch_size=batch_size,
                iterations_episode=iterations_episode)
    # obs_size = size of the state
    #obs_size = env.data_shape[1] - len(env.all_attack_names)

    # num_episodes = int(env.data_shape[0]/(iterations_episode)/10)
    num_episodes = 98  # changed from 100 to 99 since my freeuser-limit is finished on 99

    '''
    Definition for the defensor agent.
    '''
    defender_valid_actions = list(range(len(env.attack_types)))  # only detect type of attack
    defender_num_actions = len(defender_valid_actions)

    def_epsilon = 1  # exploration
    min_epsilon = 0.01  # min value for exploration
    def_gamma = 0.001
    def_decay_rate = 0.99

    def_hidden_size = 100
    def_hidden_layers = 3

    def_learning_rate = .2

    defender_agent = DefenderAgent(defender_valid_actions, obs_size, "EpsilonGreedy",
                                    epoch_length=iterations_episode,
                                    epsilon=def_epsilon,
                                    min_epsilon=min_epsilon,
                                    decay_rate=def_decay_rate,
                                    gamma=def_gamma,
                                    hidden_size=def_hidden_size,
                                    hidden_layers=def_hidden_layers,
                                    minibatch_size=minibatch_size,
                                    mem_size=1000,
                                    learning_rate=def_learning_rate,
                                    ExpRep=ExpRep)
    # Pretrained defender
    # defender_agent.model_network.model.load_weights("models/type_model.h5")



    # Statistics
    att_reward_chain = []
    def_reward_chain = []
    att_loss_chain = []
    def_loss_chain = []
    def_total_reward_chain = []
    att_total_reward_chain = []

    # Print parameters
    print("-------------------------------------------------------------------------------")
    print("Total epoch: {} | Iterations in epoch: {}"
            "| Minibatch from mem size: {} | Total Samples: {}|".format(num_episodes,
                                                                        iterations_episode, minibatch_size,
                                                                        num_episodes * iterations_episode))
    print("-------------------------------------------------------------------------------")
    print("Dataset shape: {}".format(env.data_shape))
    print("-------------------------------------------------------------------------------")
    print("Attacker parameters: Num_actions={} | gamma={} |"
            " epsilon={} | ANN hidden size={} | "
            "ANN hidden layers={}|".format(attack_num_actions,
                                            att_gamma, att_epsilon, att_hidden_size,
                                            att_hidden_layers))
    print("-------------------------------------------------------------------------------")
    print("Defense parameters: Num_actions={} | gamma={} | "
            "epsilon={} | ANN hidden size={} |"
            " ANN hidden layers={}|".format(defender_num_actions,
                                            def_gamma, def_epsilon, def_hidden_size,
                                            def_hidden_layers))
    print("-------------------------------------------------------------------------------")

    # Main loop
    attacks_by_epoch = []
    attack_labels_list = []
    for epoch in range(num_episodes):
        start_time = time.time()
        att_loss = 0.
        def_loss = 0.
        def_total_reward_by_episode = 0
        att_total_reward_by_episode = 0
        # Reset enviromet, actualize the data batch with random state/attacks
        states = env.reset()

        # Get actions for actual states following the policy
        attack_actions = attacker_agent.act(states)
        states = env.get_states(attack_actions)

        done = False

        attacks_list = []
        # Iteration in one episode
        for i_iteration in range(iterations_episode):

            attacks_list.append(attack_actions[0])
            # apply actions, get rewards and new state
            act_time = time.time()
            defender_actions = defender_agent.act(states)
            # Enviroment actuation for this actions
            next_states, def_reward, att_reward, next_attack_actions, done = env.act(defender_actions,
                                                                                        attack_actions)
            # If the epoch*batch_size*iterations_episode is largest than the df

            attacker_agent.learn(states, attack_actions, next_states, att_reward, done)
            defender_agent.learn(states, defender_actions, next_states, def_reward, done)

            act_end_time = time.time()

            # Train network, update loss after at least minibatch_learns
            if ExpRep and epoch * iterations_episode + i_iteration >= minibatch_size:
                def_loss += defender_agent.update_model()
                att_loss += attacker_agent.update_model()
            elif not ExpRep:
                def_loss += defender_agent.update_model()
                att_loss += attacker_agent.update_model()

            update_end_time = time.time()

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
        print("\r\n|Epoch {:03d}/{:03d}| time: {:2.2f}|\r\n"
                "|Def Loss {:4.4f} | Def Reward in ep {:03d}|\r\n"
                "|Att Loss {:4.4f} | Att Reward in ep {:03d}|"
                .format(epoch, num_episodes, (end_time - start_time),
                        def_loss, def_total_reward_by_episode,
                        att_loss, att_total_reward_by_episode))

        print("|Def Estimated: {}| Att Labels: {}".format(env.def_estimated_labels,
                                                            env.def_true_labels))
        attack_labels_list.append(env.def_true_labels)

        # Save the trained models
    save_model(attacker_agent, attacker_model_path)
    save_model(defender_agent, defender_model_path)
    print("Saved trained models.")

#TODO: Check if the code is working as expected
#TODO: save the models in the correct path
#TODO: load if the models are saved...
#TODO: check the logic of the code
#TODO: Add code from Jupyter notebook to the main.py (Test code and further...)