import json
import logging
import numpy as np
import tensorflow as tf
import os
import shutil
import tensorflow as tf

cwd = os.getcwd()

def logger_setup(timestamp_begin, name_suffix="default"):
    # Configure logging
    log_filename = os.path.join(cwd, f"logs/{timestamp_begin}{name_suffix}.log")
    logging.basicConfig(filename=os.path.join(cwd, log_filename), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    # Create a console handler for the logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))

    # Add the console handler to the logger
    logging.getLogger().addHandler(console_handler)

    # Redirect TensorFlow logs to the logging module
    tf.get_logger().setLevel('INFO')
    tf.get_logger().addHandler(console_handler)

    # Disable matplotlib logging
    logging.getLogger('matplotlib').setLevel(logging.WARN)

def move_log_files(current_logs_path, path):
    # Move the log files to the logs folder
    log_filename_new = f"{path}log.log"
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.copy(current_logs_path, log_filename_new)
    logging.info(f"Log file: '{current_logs_path}' \ncopied to: '{log_filename_new}'")

def log_training_parameters(num_episodes, iterations_episode, minibatch_size, env, attack_num_actions, att_gamma, att_epsilon, att_hidden_size, att_hidden_layers, att_learning_rate, defender_num_actions, def_gamma, def_epsilon, def_hidden_size, def_hidden_layers, def_learning_rate, attack_type, attack_names):
    logging.info(f"Start Training with the following parameters:\n-----------------------------------------------------------------------------------------------------------------------\n"
                 f"Total epoch: {num_episodes} | Iterations in epoch: {iterations_episode} "
                 f"| Minibatch from mem size: {minibatch_size} | Total Samples: {num_episodes * iterations_episode} | Data shape: {env.data_shape}\n"
                 f"-----------------------------------------------------------------------------------------------------------------------\n"
                 f"Attacker<Attack-Type>: Num_actions={attack_num_actions} | gamma={att_gamma} | "
                 f"epsilon={att_epsilon} | ANN hidden size={att_hidden_size} | "
                 f"ANN hidden layers={att_hidden_layers} | Learning rate={att_learning_rate}\n"
                 f"-----------------------------------------------------------------------------------------------------------------------\n"
                 f"Defender: Num_actions={defender_num_actions} | gamma={def_gamma} | "
                 f"epsilon={def_epsilon} | ANN hidden size={def_hidden_size} | "
                 f"ANN hidden layers={def_hidden_layers} | Learning rate={def_learning_rate}\n"
                 f"-----------------------------------------------------------------------------------------------------------------------\n"
                 f"Used Attack types: '{attack_type if attack_type is not None else 'all: Normal, DoS, Probe, R2L, U2R'}' | Attack name(s): {attack_names}\n"
                 f"-----------------------------------------------------------------------------------------------------------------------\n")

def print_end_of_epoch_info(epoch, num_episodes, epoch_start_time, end_time, def_loss, def_total_reward_by_episode,
                            att_loss_dos, att_total_reward_by_episode_dos, att_loss_probe, att_total_reward_by_episode_probe,
                            att_loss_r2l, att_total_reward_by_episode_r2l, att_loss_u2r, att_total_reward_by_episode_u2r,
                            env):
    logging.info(f"End of Epoch.\r\n|Epoch {epoch:03d}/{num_episodes:03d}| time: {(end_time - epoch_start_time):2.2f}|\r\n"
                 f"|Def Loss {def_loss:4.4f} | Def Reward in ep {def_total_reward_by_episode:03d}|\r\n"
                 f"|Att Loss Dos {att_loss_dos:4.4f} | Att Reward in ep {att_total_reward_by_episode_dos:03d}|\r\n"
                 f"|Att Loss Probe {att_loss_probe:4.4f} | Att Reward in ep {att_total_reward_by_episode_probe:03d}|\r\n"
                 f"|Att Loss R2L {att_loss_r2l:4.4f} | Att Reward in ep {att_total_reward_by_episode_r2l:03d}|\r\n"
                 f"|Att Loss U2R {att_loss_u2r:4.4f} | Att Reward in ep {att_total_reward_by_episode_u2r:03d}|\r\n"
                 f"|Def Estimated: {env.def_estimated_labels}| Att Labels: {env.att_true_labels}|\r\n"
                 f"|Def Amount of true predicted attacks: {env.def_true_labels}|")
    
def save_debug_info(output_dir, **kwargs):
    """Saves all relevant variables to a JSON file for debugging."""
    debug_info_path = os.path.join(output_dir, "debug_info.json")
    
    def custom_serializer(obj):
        """Custom serializer to handle non-serializable types."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    try:
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the debug information
        with open(debug_info_path, "w") as f:
            json.dump(kwargs, f, indent=4, default=custom_serializer)
        logging.info(f"Debug information saved to {debug_info_path}")
    except Exception as e:
        logging.error(f"Failed to save debug information: {e}")

def load_debug_info(debug_info_path):
    """
    Loads debug information from a JSON file.

    Args:
        debug_info_path (str): Path to the debug_info.json file.

    Returns:
        dict: A dictionary containing all the debug information.
    """
    if not os.path.exists(debug_info_path):
        raise FileNotFoundError(f"Debug info file not found: {debug_info_path}")
    
    try:
        with open(debug_info_path, "r") as f:
            debug_info = json.load(f)
        return debug_info
    except Exception as e:
        raise ValueError(f"Failed to load debug information: {e}")