import json
import logging
import numpy as np
import tensorflow as tf
import os
import shutil

from utils.config import CWD

def logger_setup(timestamp_begin, name_suffix="default"):
    # Ensure the logs directory exists
    logs_dir = os.path.join(CWD, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    # Configure logging
    log_filename = os.path.join(CWD, f"logs/{timestamp_begin}{name_suffix}.log")
    logging.basicConfig(filename=os.path.join(CWD, log_filename), level=logging.DEBUG,
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

def log_training_parameters(training_params: dict, attacker_params: dict, defender_params: dict,
                            attack_type: str, attack_names: list):
    logging.info(
        f"Start Training with the following parameters:\n"
        f"-----------------------------------------------------------------------------------------------------------------------\n"
        f"Total episodes: {training_params['num_episodes']} | Iterations in episode: {training_params['iterations_episode']} "
        f"| <Attacker> Minibatch from mem size: {training_params['minibatch_size_attacker']} | Total Samples: {training_params['total_samples']} "
        f"| <Defender> Minibatch from mem size: {training_params['minibatch_size_defender']} | Data shape: {training_params['data_shape']}\n"
        f"-----------------------------------------------------------------------------------------------------------------------\n"
        f"Attacker<Attack-Type>: Num_actions={attacker_params['num_actions']} | gamma={attacker_params['gamma']} | "
        f"epsilon={attacker_params['epsilon']} | ANN hidden size={attacker_params['hidden_size']} | "
        f"ANN hidden layers={attacker_params['hidden_layers']} | Learning rate={attacker_params['learning_rate']}\n"
        f"-----------------------------------------------------------------------------------------------------------------------\n"
        f"Defender: Num_actions={defender_params['num_actions']} | gamma={defender_params['gamma']} | "
        f"epsilon={defender_params['epsilon']} | ANN hidden size={defender_params['hidden_size']} | "
        f"ANN hidden layers={defender_params['hidden_layers']} | Learning rate={defender_params['learning_rate']}\n"
        f"-----------------------------------------------------------------------------------------------------------------------\n"
        f"Used Attack types: '{attack_type if attack_type is not None else 'all: Normal, DoS, Probe, R2L, U2R'}' | Attack name(s): {attack_names}\n"
        f"-----------------------------------------------------------------------------------------------------------------------\n"
    )

def print_end_of_epoch_info(episode_info: dict, metrics: dict, env) -> None:
    logging.info(
        f"End of Episode.\r\n|Episode {episode_info['episode']:03d}/{episode_info['num_episodes']:03d}| "
        f"time: {(episode_info['end_time'] - episode_info['epoch_start_time']):2.2f}|\r\n"
        f"|Def Loss {metrics['def_loss']:4.4f} | Def Reward in ep {metrics['def_total_reward_by_episode']:03d}|\r\n"
        f"|Att Loss Dos {metrics['att_loss_dos']:4.4f} | Att Reward in ep {metrics['att_total_reward_by_episode_dos']:03d}|\r\n"
        f"|Att Loss Probe {metrics['att_loss_probe']:4.4f} | Att Reward in ep {metrics['att_total_reward_by_episode_probe']:03d}|\r\n"
        f"|Att Loss R2L {metrics['att_loss_r2l']:4.4f} | Att Reward in ep {metrics['att_total_reward_by_episode_r2l']:03d}|\r\n"
        f"|Att Loss U2R {metrics['att_loss_u2r']:4.4f} | Att Reward in ep {metrics['att_total_reward_by_episode_u2r']:03d}|\r\n"
        f"|Def Estimated: {env.def_estimated_labels}| Att Labels: {env.att_true_labels}|\r\n"
        f"|Def Amount of true predicted attacks: {env.def_true_labels}|"
    )

def print_end_of_epoch_info_cic(episode_info: dict, metrics: dict, env) -> None:
    logging.info(
        f"End of Episode.\r\n|Episode {episode_info['episode']:03d}/{episode_info['num_episodes']:03d}| "
        f"time: {(episode_info['end_time'] - episode_info['epoch_start_time']):2.2f}|\r\n"
        f"|Def Loss {metrics['def_loss']:4.4f} | Def Reward in ep {metrics['def_total_reward_by_episode']:03d}|\r\n"
        f"|Att Loss {metrics['att_loss']:4.4f} | Att Reward in ep {metrics['att_total_reward_by_episode']:03d}|\r\n"
        f"|Def Estimated: {env.def_estimated_labels}| Att Labels: {env.att_true_labels}|\r\n"
        f"|Def Amount of true predicted attacks: {env.def_true_labels}|"
    )
    
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