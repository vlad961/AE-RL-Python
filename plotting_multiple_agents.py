from matplotlib import pyplot as plt
import logging
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf

def plot_attack_distribution_for_each_attacker(attacks_by_epoch, attack_names, path, attacker_labels=None):
    """
    Plots grouped bar charts in subplots for attack distributions of multiple attacker models over specified epochs.

    Args:
        attacks_by_epoch (list of lists): Nested list, each sub-list represents attack counts for attacker models.
                                         Format: [model_1_epoch_1, model_2_epoch_1, ...]
        attack_names (list): Dynamic list of specific attack names.
        path (str): Path to save the plots.
        attacker_labels (list): Names of attacker models (optional).
    """
    # Definiere explizit das Layout
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    #epochs_to_plot = [0, 10, 20, 30, 40, 60, 70, 80, 90]
    epochs_to_plot = [0, 5, 10, 15, 20, 30, 40, 50, 59]
    #epochs_to_plot = [0, 1, 2, 3, 5, 6, 7, 8, 9]
    num_attacks = len(attack_names)
    indices = np.arange(num_attacks)
    bar_width = 0.8 / len(attacks_by_epoch)  # Dynamisch angepasst auf Anzahl der Angreifer

    for idx, epoch in enumerate(epochs_to_plot):
        ax = axes.flatten()[idx]  # einfacher Zugriff auf die Subplots
        for attacker_idx, attacker in enumerate(attacks_by_epoch):
            ax.bar(indices + attacker_idx * bar_width,
                attacker[epoch],
                width=bar_width,
                label=attacker_labels[attacker_idx])

        ax.set_title(f'Epoch {epoch}')
        ax.set_xticks(indices + bar_width * (len(attacks_by_epoch) - 1) / 2)
        ax.set_xticklabels(attack_names, rotation=90, fontsize=7)
        ax.set_ylabel("Frequency")

    # Titel der Abbildung mit etwas Platz nach oben
    fig.suptitle("Attack distribution for each attacker", fontsize=16, y=0.95)

    # Legende etwas tiefer platzieren, direkt unter der Überschrift
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(attacks_by_epoch), 
            bbox_to_anchor=(0.5, 0.94), fontsize=10)

    # Layout optimieren, sodass die Legende Platz hat
    fig.tight_layout(rect=[0, 0, 1, 0.92])


    # Pfadprüfung und Speicherung
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(os.path.join(path, 'attack_distribution_for_each_attacker.pdf'), format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_mapped_attack_distribution_for_each_attacker(attacks_by_epoch, attack_names, path, attacker_labels=None):
    """
    Plots grouped bar charts in subplots for attack distributions of multiple attacker models over specified epochs.

    Args:
        attacks_by_epoch (list of lists): Nested list, each sub-list represents attack counts for attacker models.
                                         Format: [model_1_epoch_1, model_2_epoch_1, ...]
        attack_names (list): Names of attacks corresponding to attack indices.
        path (str): Path to save the plot.
        attacker_labels (list): Names of attacker models (optional).
    """
    num_attackers = len(attacks_by_epoch)
    num_attacks = len(attack_names)

    if attacker_labels is None:
        attacker_labels = [f'Model {i+1}' for i in range(num_attackers)]

    #epochs_to_plot = [0, 10, 20, 30, 40, 60, 70, 80, 90]
    epochs_to_plot = [0, 5, 10, 15, 20, 30, 40, 50, 59]
    #epochs_to_plot = [0, 1, 2, 3, 5, 6, 7, 8, 9]
    bar_width = 0.8 / num_attackers  # To avoid overlap

    plt.figure(figsize=(18, 18))
    plt.suptitle('Grouped Attack Distribution Across Epochs', fontsize=16)

    for indx, epoch_idx in enumerate(epochs_to_plot):
        plt.subplot(3, 3, indx + 1)
        indices = np.arange(num_attacks)

        for attacker_idx, attacker in enumerate(attacks_by_epoch):
            plt.bar(indices + attacker_idx * bar_width,
                    attacker[epoch_idx],
                    width=bar_width,
                    label=attacker_labels[attacker_idx])

        plt.xticks(indices + bar_width * (num_attackers - 1) / 2, attack_names, rotation=90)
        plt.xlabel(f"Epoch {epoch_idx}")
        plt.ylabel("Frequency")
        if indx == 0:
            plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, 'attack_distribution_mapped_for_each_attackers.pdf'), format='pdf', dpi=1000)
    plt.close()

def plot_attack_distribution_heatmap(attacks_by_epoch, attack_names, path, attacker_labels=None):
    """
    Plots dynamic heatmaps for the distribution of specific attacks across epochs for multiple attacker models.

    Args:
        attacks_by_epoch (list of lists): Nested list of shape (num_attackers, num_epochs, num_attacks).
        attack_names (list): Dynamic list of specific attack names.
        path (str): Path to save the generated heatmaps.
        attacker_labels (list): Names of the attacker models (optional).
    """
    num_attackers = len(attacks_by_epoch)

    if attacker_labels is None:
        attacker_labels = [f'Model {i+1}' for i in range(num_attackers)]

    for attacker_idx, attacker_data in enumerate(attacks_by_epoch):
        # Transpose data to have attacks as rows and epochs as columns
        data = np.array(attacker_data).T

        plt.figure(figsize=(max(14, len(data[0]) * 1.2), max(8, len(attack_names) * 0.5)))
        sns.heatmap(data, cmap='Blues', annot=True, fmt='d',
                    xticklabels=[f'Epoch {i}' for i in range(data.shape[1])],
                    yticklabels=attack_names)

        plt.title(f'Attack Distribution Heatmap - {attacker_labels[attacker_idx]}', fontsize=16)
        plt.xlabel('Epochs')
        plt.ylabel('Attack Types')

        plt.tight_layout()

        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(os.path.join(path, f'attack_distribution_heatmap_{attacker_labels[attacker_idx]}.pdf'),
                    format='pdf', dpi=1000)
        plt.close()

def plot_rewards_losses_boxplot(reward_chains, loss_chains, model_labels, path):
    """
    Plots boxplots for reward and loss distributions across episodes for multiple models.

    Args:
        reward_chains (list of lists): Rewards per episode for each model.
        loss_chains (list of lists): Losses per episode for each model.
        model_labels (list): Labels for the models (e.g., attackers and defenders).
        path (str): Path to save the plots.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Preparing data for rewards
    reward_data = pd.DataFrame({
        label: rewards for label, rewards in zip(model_labels, reward_chains)
    })

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=reward_data)
    plt.title('Distribution of Rewards per Episode')
    plt.xlabel('Model')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'rewards_boxplot.pdf'), format='pdf', dpi=1000)
    plt.close()

    # Preparing data for losses
    loss_data = pd.DataFrame({
        label: losses for label, losses in zip(model_labels, loss_chains)
    })

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=loss_data)
    plt.title('Distribution of Losses per Episode')
    plt.xlabel('Model')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'losses_boxplot.pdf'), format='pdf', dpi=1000)
    plt.close()

def plot_trend_lines(metrics_by_model, model_labels, metric_name, path):
    """
    Plots linear trend lines of a specific metric (e.g., rewards or losses) over episodes for each model separately.

    Args:
        metrics_by_model (list of lists): Metric values per episode for each model.
        model_labels (list): Labels for the models (e.g., attackers and defenders).
        metric_name (str): Name of the metric (e.g., 'Reward' or 'Loss').
        path (str): Path to save the generated plots.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    episodes = range(1, len(metrics_by_model[0]) + 1)

    for idx, metrics in enumerate(metrics_by_model):
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, metrics, marker='o', linestyle='-', label=f'{model_labels[idx]}')
        plt.title(f'{metric_name} Trend Over Episodes - {model_labels[idx]}')
        plt.xlabel('Episode')
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'{metric_name.lower()}_trend_{model_labels[idx].replace(" ", "_").lower()}.pdf'), format='pdf', dpi=1000)
        plt.close()

def plot_trend_lines2(metrics_rewards, metrics_losses, model_label, path):
    """
    Plots rewards and losses in one figure for direct comparison per model.

    Args:
        metrics_rewards (list): Reward values per episode for the model.
        metrics_losses (list): Loss values per episode for the model.
        model_label (str): Label of the model (e.g., attacker or defender).
        path (str): Path to save the generated plots.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    num_entries = len(metrics_rewards)
    if num_entries > 50:
        step = 10
    elif num_entries >= 20:
        step = 5
    else:
        step = 1

    x_ticks = np.arange(0, num_entries, step)
    x_labels = [f'{i}' for i in x_ticks]

    plt.figure(figsize=(10, 8))

    # Reward subplot
    plt.subplot(211)
    plt.plot(np.arange(num_entries), metrics_rewards, label=f'{model_label} Reward', color='tab:blue')
    plt.title(f'Total Reward by Episode - {model_label}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.xticks(x_ticks, x_labels)
    plt.legend(loc='upper right')

    # Loss subplot
    plt.subplot(212)
    plt.plot(np.arange(num_entries), metrics_losses, label=f'{model_label} Loss', color='tab:orange')
    plt.title(f'Loss by Episode - {model_label}')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.xticks(x_ticks, x_labels)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'reward_loss_trend_{model_label.replace(" ", "_").lower()}.pdf'), format='pdf', dpi=1000)
    plt.close()

def plot_trend_lines_multiple_agents(rewards_list, losses_list, model_labels, path):
    """
    Plots rewards and losses trends for multiple models (agents) in two subplots.

    Args:
        rewards_list (list of lists): Reward values per episode for each model.
        losses_list (list of lists): Loss values per episode for each model.
        model_labels (list): Labels for each model (agents).
        path (str): Path to save the generated plots.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    num_entries = len(rewards_list[0])
    if num_entries > 50:
        step = 10
    elif num_entries >= 20:
        step = 5
    else:
        step = 1

    x_ticks = np.arange(0, num_entries, step)
    x_labels = [f'{i}' for i in x_ticks]

    plt.figure(figsize=(12, 10))

    # Reward subplot
    plt.subplot(211)
    for idx, rewards in enumerate(rewards_list):
        plt.plot(np.arange(num_entries), rewards, label=f'{model_labels[idx]} Reward')
    plt.title('Total Rewards by Episode (All Agents)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.xticks(x_ticks, x_labels)
    plt.legend(loc='upper right')

    # Loss subplot
    plt.subplot(212)
    for idx, losses in enumerate(losses_list):
        plt.plot(np.arange(num_entries), losses, label=f'{model_labels[idx]} Loss')
    plt.title('Losses by Episode (All Agents)')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.xticks(x_ticks, x_labels)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'reward_loss_trend_all_agents.pdf'), format='pdf', dpi=1000)
    plt.close()

def plot_trends_heatmap(metrics_list, model_labels, metric_name, path):
    """
    Plots a heatmap to visualize trends (e.g., rewards or losses) across episodes for multiple models.

    Args:
        metrics_list (list of lists): Metric values per episode for each model.
        model_labels (list): Labels for each model (agents).
        metric_name (str): Name of the metric (e.g., 'Reward', 'Loss').
        path (str): Path to save the generated plot.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    data = np.array(metrics_list)

    plt.figure(figsize=(15, len(model_labels) * 1.5))
    sns.heatmap(data, cmap='Blues', annot=True, fmt='.2f', linewidths=.5,
                xticklabels=[f'Ep {i+1}' for i in range(data.shape[1])],
                yticklabels=model_labels)

    plt.title(f'{metric_name} Trends Heatmap Across Episodes', fontsize=16)
    plt.xlabel('Episode')
    plt.ylabel('Model')

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{metric_name.lower()}_trends_heatmap.pdf'), format='pdf', dpi=1000)
    plt.close()

def plot_attack_distributions_multiple_agents(attacks_by_epoch, attack_map, attack_names, attacks_mapped_to_att_type_list, path) -> plt.Figure:
    # Create a mapping from attack id to index of attack_names
    attack_id_to_index = {idx: attack_names.index(label) for idx, label in enumerate(attack_map.keys()) if label in attack_names}
    
    bins=np.arange(len(attack_names) + 1)
    # Plot attacks distribution alongside
    plt.figure(2,figsize=[12,12])
    plt.xticks([])
    plt.yticks([])
    plt.title("Attacks distribution throughout episodes")
    for indx,e in enumerate([0, 5, 10, 15, 20, 30, 40, 50, 59]):
    #for indx,e in enumerate([0, 1, 2, 3, 5, 6, 7, 8, 9]):
        flattened_epoch = [int(action[0]) for actions in attacks_by_epoch[e] for action in actions]
        
        # Map attack ids to indices in attack_names
        mapped_epoch = [attack_id_to_index.get(value, -1) for value in flattened_epoch]
        mapped_epoch = [value for value in mapped_epoch if value != -1]  # Entferne ungültige Werte
        
        plt.subplot(3,3,indx+1)
        #plt.hist(flattened_epoch, bins=bins, width=0.9, align='left')
        counts, _, bars = plt.hist(mapped_epoch, bins=bins, width=0.9, align='left', color='skyblue', edgecolor='black')
        plt.xlabel("{} epoch".format(e))
        plt.xticks(bins[:-1], attack_names, rotation=90)

        # Set Y-Axis to integer ticks only
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        # Dynamic y-axis limit
        max_height = max(counts) if len(counts) > 0 else 0
        plt.ylim(0, max_height + max_height * 0.1)  # 10% buffer to top

        # Add counts to the bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,  # Position over the bar
                     f'{int(count)}', ha='center', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'attacks_distribution.pdf'), format='pdf', dpi=1000)
    plt.close()

    # Plot attacks distribution alongside
    plt.figure(3,figsize=[10,10])
    plt.xticks([])
    plt.yticks([])
    plt.title("Attacks (mapped) distribution throughout  episodes")
    for indx,e in enumerate([0, 5, 10, 15, 20, 30, 40, 50, 59]):
    #for indx,e in enumerate([0, 1, 2, 3, 5, 6, 7, 8, 9]):
        plt.subplot(3,3,indx+1)
        #plt.bar(range(5), attacks_mapped_to_att_type_list[e], tick_label = ['Normal','Dos','Probe','R2L','U2R'])
        bars = plt.bar(range(5), attacks_mapped_to_att_type_list[e], tick_label=['Normal', 'Dos', 'Probe', 'R2L', 'U2R'], color='skyblue', edgecolor='black')
        plt.xlabel("{} epoch".format(e))

        # Set Y-Axis to integer ticks only
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        # Dynamic y-axis limit
        max_height = max([bar.get_height() for bar in bars]) if len(bars) > 0 else 0
        plt.ylim(0, max_height + max_height * 0.1)  # 10% buffer to top

        # Add counts to the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,  # Position over the bar
                     f'{int(height)}', ha='center', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'attacks_mapped_distribution.pdf'), format='pdf', dpi=1000)
    plt.close()

    logging.info(f"Plotted attack distributions during training in: {path}")

    return plt

def plot_rewards_and_losses_during_training_multiple_agents(def_reward_chain, att_reward_chain, def_loss_chain, att_loss_chain, path) -> plt.Figure:
    """
    Plot the results of the training process
    Args:
        def_reward_chain: list of rewards of the defense agent
        att_reward_chain: list of rewards of the attack agent
        def_loss_chain: list of losses of the defense agent
        att_loss_chain: list of losses of the attack agent
    Returns:
        plt: the matplotlib.pyplot module for further manipulation
    """

    if not os.path.exists(path):
        os.makedirs(path)
    # Plot training results

    # Create x-ticks based on the length of def_reward_chain
    num_entries = len(def_reward_chain)
    if num_entries > 50:
        step = 10
    elif num_entries >= 20:
        step = 5
    else:
        step = 1

    x_ticks = np.arange(0, num_entries, step)
    x_labels = [f'{i}' for i in x_ticks]

    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.arange(len(def_reward_chain)),def_reward_chain,label='Defense')
    plt.plot(np.arange(len(att_reward_chain)),att_reward_chain,label='Attack')
    plt.title('Total reward by episode')
    plt.xlabel('n Episode')
    plt.ylabel('Total reward')
    plt.xticks(x_ticks, x_labels)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=2, mode="expand", borderaxespad=0.)

    plt.subplot(212)
    plt.plot(np.arange(len(def_loss_chain)), def_loss_chain,label='Defense')
    plt.plot(np.arange(len(att_loss_chain)), att_loss_chain,label='Attack (Aggregated sum of losses over all Agents)')
    plt.title('Loss by episode')
    plt.xlabel('n Episode')
    plt.ylabel('loss')
    plt.xticks(x_ticks, x_labels)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=2, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'train_reward_loss.pdf'), format='pdf', dpi=1000)
    plt.close()

    logging.info(f"Plotted rewards and losses during training in: {path}")
    return plt

def plot_training_metrics(history, save_path=None):
    epochs = range(1, len(history.history['mse']) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history.history['mse'], label='MSE')
    plt.xlabel('Episoden')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE über Episoden')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(f"{save_path}/mse_over_episodes.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history.history['mae'], label='MAE', color='orange')
    plt.xlabel('Episoden')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('MAE über Episoden')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(f"{save_path}/mae_ueber_episoden.png")
    plt.close()

def plot_training_error(mse_before, mae_before, mse_after, mae_after, save_path=None):
    epochs = range(1, len(mse_before) + 1)

    plt.figure(figsize=(14, 7))

    # Plot für MSE
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mse_before, label='MSE before Update (implicit)', marker='o')
    plt.plot(epochs, mse_after, label='MSE after Update (explicit)', marker='x', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('MSE')
    plt.title('MSE-Trend before and after Update')
    plt.grid(True)
    plt.legend()

    # Plot für MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mae_before, label='MAE before Update (implicit)', marker='o')
    plt.plot(epochs, mae_after, label='MAE after Update (explicit)', marker='x', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('MAE')
    plt.title('MAE-Trend before and after Update')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/mse_mae_trends.pdf", format='pdf', dpi=1000)
    plt.close()

def visualize_q_value_errors(model, states, target_q_values, plots_path=None):
    # Vorhersage der Q-Werte
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    predicted_q_values = model.predict(states_tensor)

    # Berechnung der Fehler-Metriken
    mse = np.mean((predicted_q_values - target_q_values)**2)
    mae = np.mean(np.abs(predicted_q_values - target_q_values))

    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")

    # 1. Histogramm der absoluten Fehler
    absolute_errors = np.abs(predicted_q_values - target_q_values)
    plt.figure(figsize=(10, 6))
    plt.hist(absolute_errors.flatten(), bins=50, color='skyblue', alpha=0.7)
    plt.xlabel('Absoluter Fehler')
    plt.ylabel('Häufigkeit')
    plt.title('Histogramm der absoluten Fehler (MAE)')
    plt.grid(True)
    if plots_path:
        plt.savefig(f"{plots_path}/histogram_mae.png")
    plt.close()

    # 2. Scatterplot: Vorhergesagte vs. Wahre Q-Werte
    plt.figure(figsize=(8, 8))
    plt.scatter(target_q_values.flatten(), predicted_q_values.flatten(), alpha=0.4)
    plt.plot([target_q_values.min(), target_q_values.max()],
             [target_q_values.min(), target_q_values.max()], 'k--', lw=2, color='red', label="ideale Vorhersage")
    plt.xlabel('Wahre Q-Werte')
    plt.ylabel('Vorhergesagte Q-Werte')
    plt.title('Scatterplot der vorhergesagten vs. wahren Q-Werte')
    plt.legend(['Ideale Vorhersage', 'Tatsächliche Werte'])
    plt.grid(True)
    if plots_path:
        plt.savefig(f"{plots_path}/scatter_q_values.png")
    plt.close()

