from typing import Union
from data.cic_data_manager import CICDataManager
from data.nsl_kdd_data_manager import NslKddDataManager
from models.attack_agent import AttackAgent
from models.defender_agent import DefenderAgent
from utils.helpers import create_attack_id_to_index_mapping, create_attack_id_to_type_mapping, transform_attacks_by_epoch, transform_attacks_by_type
from utils.plotting_multiple_agents import plot_attack_distribution_for_each_attacker, plot_attack_distributions_multiple_agents, plot_mapped_attack_distribution_for_each_attacker, plot_rewards_and_losses_during_training_multiple_agents, plot_rewards_losses_boxplot, plot_training_error, plot_trend_lines_multiple_agents


def plot_training_diagrams(
    multiple_attackers, 
    attack_indices_per_episode, 
    cic_attack_map, 
    data_mgr: Union [NslKddDataManager | CICDataManager], 
    attacks_mapped_to_att_type_list, 
    plots_path, 
    rewards, 
    losses, 
    is_cic_2017_trainingset=False, agent_defender: DefenderAgent =None,
    attackers: list[AttackAgent]=None,
    mse_before_history=None, mae_before_history=None):

    plot_trend_lines_multiple_agents(rewards, losses, [agent_defender.name] + [attacker.name for attacker in attackers], plots_path)
    plot_rewards_losses_boxplot(rewards, losses, [agent_defender.name] + [attacker.name for attacker in attackers], plots_path)
    plot_training_error(mse_before_history, mae_before_history, save_path=plots_path)

    if multiple_attackers:
        plot_rewards_and_losses_during_training_multiple_agents(rewards[0], rewards[1], losses[0], losses[1], plots_path) # Flag setzen
        plot_attack_distributions_multiple_agents(attack_indices_per_episode, cic_attack_map, data_mgr.attack_names, attacks_mapped_to_att_type_list, plots_path)
        
        attack_id_to_index = create_attack_id_to_index_mapping(data_mgr.attack_map, data_mgr.attack_names)
        num_attacks = len(data_mgr.attack_names)
        transformed_attacks = transform_attacks_by_epoch(attack_indices_per_episode, attack_id_to_index, num_attacks)
        plot_attack_distribution_for_each_attacker(transformed_attacks, data_mgr.attack_names, plots_path, [attacker.name for attacker in attackers])
        
        attack_id_to_type = create_attack_id_to_type_mapping(data_mgr.attack_map)
        transformed_attacks_by_type = transform_attacks_by_type(attack_indices_per_episode, attack_id_to_type, data_mgr.attack_types)
        plot_mapped_attack_distribution_for_each_attacker(transformed_attacks_by_type, data_mgr.attack_types, plots_path, [attacker.name for attacker in attackers])

    if not multiple_attackers:
        plot_attack_distributions_multiple_agents(
            attack_indices_per_episode, cic_attack_map, data_mgr.attack_names,
            attacks_mapped_to_att_type_list, plots_path,
            attack_type=data_mgr.attack_types, use_direct_name_mapping=True
        )