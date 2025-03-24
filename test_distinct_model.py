import os
from log_config import load_debug_info, logger_setup
from models.helpers import create_attack_id_to_index_mapping, transform_attacks_by_epoch
from plotting_multiple_agents import plot_attack_distribution_for_each_attacker, plot_attack_distributions_multiple_agents, plot_rewards_and_losses_during_training_multiple_agents, plot_rewards_losses_boxplot, plot_training_error, plot_trend_lines_multiple_agents
from test_multiple_agents import test_trained_agent_quality
from datetime import datetime
from data.data_cls import attack_map

cwd = os.getcwd()
defender_model_path = os.path.join(cwd, "models/trained-models/2025-03-24-10-22-WIN-multiple-attackers-att-5L-def-3L-lr-0.001/defender_model.keras")
plots_path = os.path.join(cwd, "models/trained-models/2025-03-24-10-22-WIN-multiple-attackers-att-5L-def-3L-lr-0.001/plots/")
destination_log_path = os.path.join(cwd, "models/trained-models/2025-03-24-10-22-WIN-multiple-attackers-att-5L-def-3L-lr-0.001/logs/")

if __name__ == "__main__":
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger_setup(timestamp_begin)


    # Pfad zur debug_info.json-Datei
    debug_info_path = os.path.join(destination_log_path, "debug_info.json")

    # Debug-Informationen laden
    debug_info = load_debug_info(debug_info_path)

    # Zugriff auf die geladenen Variablen
    rewards = debug_info["rewards"]
    losses = debug_info["losses"]
    attack_indices_per_episode = debug_info["attack_indices_per_episode"]
    attack_names_per_episode = debug_info["attack_names_per_episode"]
    attacks_mapped_to_att_type_list = debug_info["attacks_mapped_to_att_type_list"]
    mse_before_history = debug_info["mse_before_history"]
    mae_before_history = debug_info["mae_before_history"]
    mse_after_history = debug_info["mse_after_history"]
    mae_after_history = debug_info["mae_after_history"]
    agents = debug_info["agents"]
    attack_id_to_index = debug_info["attack_id_to_index"]
    attack_id_to_type = debug_info["attack_id_to_type"]
    attack_names = debug_info["attack_names"]
    attack_types = debug_info["attack_types"]
    plots_path = debug_info["plots_path"]

    # Test- und Visualisierungs-Code ausführen
    plot_rewards_and_losses_during_training_multiple_agents(
        rewards["defender"], 
        [rewards["dos"], rewards["probe"], rewards["r2l"], rewards["u2r"]], 
        losses["defender"], 
        [losses["dos"], losses["probe"], losses["r2l"], losses["u2r"]], 
        plots_path
    )

    plot_attack_distributions_multiple_agents(
        attack_indices_per_episode, 
        attack_id_to_type, 
        attack_names, 
        attacks_mapped_to_att_type_list, 
        plots_path
    )

    plot_trend_lines_multiple_agents(rewards, losses, ["Defender", "Attacker: DoS", "Attacker: Probe", "Attacker: R2L", "Attacker: U2R"], plots_path)

    attack_id_to_index = create_attack_id_to_index_mapping(attack_map, attack_names)
    num_attacks = len(attack_names)
    transformed_attacks = transform_attacks_by_epoch(attack_indices_per_episode, attack_id_to_index, num_attacks)
    plot_attack_distribution_for_each_attacker(transformed_attacks, attack_names, plots_path, ['Attacker DoS', 'Attacker Probe', 'Attacker R2L', 'Attacker U2R'])
    plot_rewards_losses_boxplot(rewards, losses, ["Defender", "Attacker: DoS", "Attacker: Probe", "Attacker: R2L", "Attacker: U2R"], plots_path)# FIXME: überprüfe ob Logik korrekt ist. 
    plot_training_error(
        mse_before_history, 
        mae_before_history, 
        mse_after_history, 
        mae_after_history, 
        save_path=plots_path
    )


    test_trained_agent_quality(defender_model_path, plots_path)



