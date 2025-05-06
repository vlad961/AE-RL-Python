from matplotlib import pyplot as plt
import logging
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd
import seaborn as sns

from utils.plotting_multiple_agents import select_epochs_to_plot

cwd = os.getcwd()
model_comparison_path = os.path.join(cwd, "model-comparison")

######################
# Plotting functions #
######################
def plot_rewards_and_losses_during_training(def_reward_chain, att_reward_chain, def_loss_chain, att_loss_chain, path) -> plt.Figure:
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
    plt.plot(np.arange(len(att_loss_chain)), att_loss_chain,label='Attack')
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

def plot_attack_distributions(attacks_by_epoch, attack_names, attacks_mapped_to_att_type_list, path) -> plt.Figure:
    bins=np.arange(len(attack_names) + 1 )
    # Plot attacks distribution alongside
    plt.figure(2,figsize=[12,12])
    plt.xticks([])
    plt.yticks([])
    plt.title("Attacks distribution throughout episodes")
    epochs = select_epochs_to_plot(attacks_by_epoch)
    for indx, e in enumerate(epochs):
        plt.subplot(3, 3, indx + 1)

        # Flatten falls verschachtelte Struktur
        epoch_data = attacks_by_epoch[e]
        flattened_epoch = [int(action[0]) if isinstance(action, (list, tuple)) else int(action)
                           for action in epoch_data]

        counts, _, bars = plt.hist(flattened_epoch, bins=bins, width=0.9, align='left', color='skyblue', edgecolor='black')
        plt.xlabel(f"{e} epoch")
        plt.xticks(bins[:-1], attack_names, rotation=90)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        max_height = max(counts) if len(counts) > 0 else 0
        plt.ylim(0, max_height + max_height * 0.1)

        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{int(count)}', ha='center', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'attacks_distribution.pdf'), format='pdf', dpi=1000)
    plt.close()

    # Plot attacks distribution alongside
    plt.figure(3,figsize=[10,10])
    plt.xticks([])
    plt.yticks([])
    plt.title("Attacks (mapped) distribution throughout  episodes")
    for indx,e in enumerate(epochs):
        plt.subplot(3,3,indx+1)
        plt.bar(range(5), attacks_mapped_to_att_type_list[e], tick_label = ['Normal','Dos','Probe','R2L','U2R'])
        plt.xlabel("{} epoch".format(e))

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'attacks_mapped_distribution.pdf'), format='pdf', dpi=1000)
    plt.close()

    logging.info(f"Plotted attack distributions during training in: {path}")

    return plt

def plot_confusion_matrix(cm, classes,
                          path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues) -> plt.Figure:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Normalized confusion matrix")
    else:
        logging.info('Confusion matrix, without normalization')


    logging.info(f"\nConfusion Matrix:\n{cm}")
    fig_height = max(9, 0.6 * len(classes))
    fig_width = max(10, 0.6 * len(classes))
    fmt = '.4f' if normalize else 'd'
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, 'confusion_matrix.pdf'), format='pdf', dpi=1000)
    plt.close()

    logging.info(f"Plotted confusion matrix in: {path}")
    return plt

def plot_line_diagram(df, title="Accuracy über verschiedene Learning Rates", x_label="Learning rate", y_label="Accuracy"):
    # Liniendiagramm erstellen
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='LR', y='value', hue='variable', data=pd.melt(df, ['LR']))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    #plt.xscale('log')  # Logarithmische Skala für die x-Achse
    plt.grid(True)
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, 'line_diagram.pdf'), format='pdf', dpi=1000)
    plt.close()


#import matplotlib.pyplot as plt
#import numpy as np

def plot_metrics_comparison_of_original_and_my_code():
    # Daten für Original Code (1 HL, 5 HL) und My Code (5 HL)
    categories = ["Accuracy", "F1", "Precision", "Recall"]
    original_1HL = [0.8156, 0.8058, 0.8112, 0.8156]  # Original Code (1 HL, 3 HL, LR=0.00025)
    original_5HL = [0.7868, 0.7805, 0.786, 0.7868]  # Original Code (5 HL, 3 HL, LR=0.00025)
    my_code_5HL = [0.8130, 0.8019, 0.8050, 0.8130]  # My Code (5 HL, 3 HL, LR=0.001)

    # Balkenbreite und Position
    bar_width = 0.25
    x = np.arange(len(categories))

    # Diagramm erstellen
    fig, ax = plt.subplots(figsize=(8, 6))

    # Balken zeichnen
    bars1 = ax.bar(x - bar_width, original_1HL, bar_width, label="Original (Att=1HL, Def=3HL, LR=0.00025)", alpha=0.8)
    bars2 = ax.bar(x, original_5HL, bar_width, label="Original (Att=5HL, Def=3HL, LR=0.00025)", alpha=0.8)
    bars3 = ax.bar(x + bar_width, my_code_5HL, bar_width, label="My Code (5HL, LR=0.001)", alpha=0.8)

    # Werte in Balken einfügen
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Achsentitel und Labels
    ax.set_ylabel("Score")
    ax.set_title("Vergleich der Metriken zwischen Original und My Code")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    
    # Legende weiter unten platzieren
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

    # Diagramm anzeigen und speichern
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "comparison_metrics.pdf"), format='pdf', dpi=1000)
    plt.close()


def plot_line_diagram_to_compare_accuracy_of_various_models():
    # Beispiel-Daten (ersetze dies durch deine tatsächlichen Daten)
    data = {
        'Modell': ['Original Code', 'My Code', 'Attacker = 5 HL'],
        'Accuracy': [0.8156, 0.8345, 0.8212]
    }
    df = pd.DataFrame(data)

    # Balkendiagramm erstellen
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Modell', y='Accuracy', data=df)
    plt.xlabel('Modell')
    plt.ylabel('Accuracy')
    plt.title('Vergleich der Accuracy verschiedener Modelle')
    plt.xticks(rotation=45)  # Drehe die x-Achsenbeschriftungen für bessere Lesbarkeit
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "comparison_models_accuracy.pdf"), format='pdf', dpi=1000)
    plt.close()

def plot_boxplot():
    # Beispiel-Daten (ersetze dies durch deine tatsächlichen Daten)
    data = {
        'Modell': ['Original Code'] * 3 + ['My Code'] * 3,
        'Accuracy': [0.8495, 0.8117, 0.7857, 0.8508, 0.8223, 0.7994]
    }
    df = pd.DataFrame(data)

    # Boxplot erstellen
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Modell', y='Accuracy', data=df)
    plt.xlabel('Modell')
    plt.ylabel('Accuracy')
    plt.title('Verteilung der Accuracy für verschiedene Modelle')
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "boxplot_models_accuracy.pdf"), format='pdf', dpi=1000)
    plt.close()

def plot_heatmap_for_configs(): # TODO: sieht gut aus für eine große Anzahl von Modellen und Metriken
    # Beispiel-Daten (ersetze dies durch deine tatsächlichen Daten)
    data = {
        'Modell': ['Modell A', 'Modell B', 'Modell C'],
        'Accuracy': [0.85, 0.82, 0.88],
        'F1-Score': [0.83, 0.80, 0.86],
        'Precision': [0.86, 0.84, 0.89],
        'Recall': [0.84, 0.78, 0.87]
    }
    df = pd.DataFrame(data).set_index('Modell')

    # Heatmap erstellen
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='viridis')
    plt.title('Performance verschiedener Modelle über verschiedene Metriken')
    plt.xlabel('Metriken')
    plt.ylabel('Modelle')
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "heatmap_models_metrics.pdf"), format='pdf', dpi=1000)
    plt.close()

def plot_accuracy_vs_hidden_layers(): # //TODO: sieht gut aus für eine kleine Anzahl von Modellen und Metriken
    # Beispiel-Daten (ersetze dies durch deine tatsächlichen Daten)
    data = {
        'Hidden Layers': [1, 3, 5, 8],
        'Accuracy': [0.82, 0.85, 0.87, 0.86]
    }
    df = pd.DataFrame(data)

    # Streudiagramm erstellen
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Hidden Layers', y='Accuracy', data=df)
    plt.title('Accuracy in Abhängigkeit von der Anzahl der Hidden Layers')
    plt.xlabel('Anzahl der Hidden Layers')
    plt.ylabel('Accuracy')
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "scatterplot_accuracy_hidden_layers.pdf"), format='pdf', dpi=1000)
    plt.close()

def plot_accuracy_vs_learning_rate():
    # Beispiel-Daten (ersetze dies durch deine tatsächlichen Daten)
    data = {
        'Learning Rate': [0.00025, 0.001, 0.01],
        'Accuracy': [0.85, 0.87, 0.82]
    }
    df = pd.DataFrame(data)

    # Streudiagramm erstellen
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Learning Rate', y='Accuracy', data=df)
    plt.title('Accuracy in Abhängigkeit von der Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    #plt.xscale('log')  # Logarithmische Skala für die x-Achse
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "scatterplot_accuracy_learning_rate.pdf"), format='pdf', dpi=1000)
    plt.close()

def plot_confusion_matrix_comparison():
    # Daten für Original Code (1 HL, 5 HL) und My Code (5 HL)
    categories = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    best_run = [0.9502, 0.8489, 0.8207, 0.2865, 0.2700]  # Original Code (1 HL, 3 HL, LR=0.00025)
    only_attacks = [0.0, 0.8968, 0.7964, 0.7447, 0.2300]  # Original Code (5 HL, 3 HL, LR=0.00025)
    normal_and_x = [0.9800, 0.8423, 0.8145, 0.3072, 0.2250]  # My Code (5 HL, 3 HL, LR=0.001)

    # Balkenbreite und Position
    bar_width = 0.25
    x = np.arange(len(categories))

    # Diagramm erstellen
    fig, ax = plt.subplots(figsize=(8, 6))

    # Balken zeichnen
    bars1 = ax.bar(x - bar_width, best_run, bar_width, label="All (Att=5HL, Def=3HL, LR=0.001)", alpha=0.8)
    bars2 = ax.bar(x, only_attacks, bar_width, label="Only Attacks (Att=5HL, Def=3HL, LR=0.001)", alpha=0.8)
    bars3 = ax.bar(x + bar_width, normal_and_x, bar_width, label="Normal and X (Att=5HL, Def=3HL, LR=0.001)", alpha=0.8)

    # Werte in Balken einfügen
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=7)

    # Achsentitel und Labels
    ax.set_ylabel("Score")
    ax.set_title("Vergleich von 'All', 'Attacks' und 'Normal & Klasse-X'")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    
    # Legende weiter unten platzieren
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

    # Diagramm anzeigen und speichern
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "comparison_confusion_matrix_between_different_models.pdf"), format='pdf', dpi=1000)
    plt.close()

def plot_all_only_attacks_one_vs_all_metrics2():
    # Definieren der Tabellen
    result_model_trained_only_on_attacks = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.569242, 0.615907, 0.890924, 0.897534, 0.976801],
        "f1": [0.0, 0.607034, 0.62521, 0.625365, 0.149593],
        "precision": [0.0, 0.458805, 0.538744, 0.51482, 0.110843],
        "recall": [0.0, 0.896755, 0.744735, 0.796365, 0.23]
    })

    result_model_trained_on_all_types = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.880767, 0.935504, 0.874867, 0.959812, 0.98035],
        "f1": [0.87286, 0.896996, 0.358718, 0.814344, 0.196007],
        "precision": [0.807191, 0.950886, 0.479635, 0.808052, 0.153846],
        "recall": [0.95016, 0.848887, 0.286492, 0.820735, 0.27]
    })

    # Zusammenführen der F1-Werte beider Tabellen
    comparison_df = result_model_trained_only_on_attacks[["name", "f1"]].rename(columns={"f1": "f1_model_1"}).merge(
        result_model_trained_on_all_types[["name", "f1"]].rename(columns={"f1": "f1_model_2"}), on="name"
    )

    # Erstellen eines Balkendiagramms zum Vergleich der F1-Werte
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    x_labels = comparison_df["name"]
    x = range(len(x_labels))

    plt.bar(x, comparison_df["f1_model_1"], width=bar_width, label="F1 - Model Only Attacks")
    plt.bar([p + bar_width for p in x], comparison_df["f1_model_2"], width=bar_width, label="F1 - Model All Types")

    plt.xticks([p + bar_width / 2 for p in x], x_labels)
    plt.ylabel("F1-Score")
    plt.title("Vergleich der F1-Scores zwischen zwei Modellen")
    plt.legend()
    plt.savefig(os.path.join(model_comparison_path, "comparison_one_vs_all_metrics.pdf"), format='pdf', dpi=1000)
    plt.close()

def plot_one_vs_all_metrics_comparison_between_all_only_attacks_and_normal_and_class():
    # Neuimport nach Reset
    # Definieren der Tabellen
    only_attacks = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.0, 0.615907, 0.890924, 0.897534, 0.976801],
        "f1": [0.0, 0.607034, 0.62521, 0.625365, 0.149593],
        "precision": [0.0, 0.458805, 0.538744, 0.51482, 0.110843],
        "recall": [0.0, 0.896755, 0.744735, 0.796365, 0.23]
    })

    all_types = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.880767, 0.935504, 0.874867, 0.959812, 0.98035],
        "f1": [0.87286, 0.896996, 0.358718, 0.814344, 0.196007],
        "precision": [0.807191, 0.950886, 0.479635, 0.808052, 0.153846],
        "recall": [0.95016, 0.848887, 0.286492, 0.820735, 0.27]
    })

    two_classes = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.0, 0.870742, 0.878948, 0.730882, 0.96305],
        "f1": [0.0, 0.811733, 0.382719, 0.393967, 0.110276],
        "precision": [0.0, 0.783292, 0.507499, 0.259816, 0.062241],
        "recall": [0.0, 0.842317, 0.30719, 0.814539, 0.225]
    })

    # Kombinieren der drei Modelle für alle Metriken
    metrics = ["accuracy", "f1", "precision", "recall"]
    comparison_dfs = {}

    for metric in metrics:
        comparison_dfs[metric] = only_attacks[["name", metric]].rename(columns={metric: f"{metric}_model_1"}).merge(
            all_types[["name", metric]].rename(columns={metric: f"{metric}_model_2"}), on="name"
        ).merge(
            two_classes[["name", metric]].rename(columns={metric: f"{metric}_model_3"}), on="name"
        )

    # Visualisierung aller Metriken
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bar_width = 0.3
    x_labels = only_attacks["name"]
    x = range(len(x_labels))

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df_metric = comparison_dfs[metric]

        bars1 = ax.bar(x, df_metric[f"{metric}_model_1"], width=bar_width, label=f"{metric} - Model Only Attacks")
        bars2 = ax.bar([p + bar_width for p in x], df_metric[f"{metric}_model_2"], width=bar_width, label=f"{metric} - Model All Types")
        bars3 = ax.bar([p + 2 * bar_width for p in x], df_metric[f"{metric}_model_3"], width=bar_width, label=f"{metric} - Model Normal & Klasse-X")

            # Werte über die Balken schreiben
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks([p + bar_width for p in x])
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Vergleich der One-vs-All {metric.capitalize()}-Werte zwischen den drei Modellen")
        ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "comparison_one_vs_all_metrics.pdf"), format='pdf', dpi=1000)   
    plt.close()


def plot_overall_metrics_comparison_between_all_only_attacks_and_normal_and_class():
    # Neuimport nach Reset
    # Definieren der Tabellen
    only_attacks = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.0, 0.615907, 0.890924, 0.897534, 0.976801],
        "f1": [0.0, 0.607034, 0.62521, 0.625365, 0.149593],
        "precision": [0.0, 0.458805, 0.538744, 0.51482, 0.110843],
        "recall": [0.0, 0.896755, 0.744735, 0.796365, 0.23]
    })

    all_types = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.880767, 0.935504, 0.874867, 0.959812, 0.98035],
        "f1": [0.87286, 0.896996, 0.358718, 0.814344, 0.196007],
        "precision": [0.807191, 0.950886, 0.479635, 0.808052, 0.153846],
        "recall": [0.95016, 0.848887, 0.286492, 0.820735, 0.27]
    })

    two_classes = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.0, 0.7007, 0.4658, 0.4853, 0.4313],
        "f1": [0.0, 0.6067, 0.3187, 0.3555, 0.2644],
        "precision": [0.0, 0.5413, 0.2612, 0.2862, 0.1907],
        "recall": [0.0, 0.7007, 0.4658, 0.4853, 0.4313]
    })

    # Kombinieren der drei Modelle für alle Metriken
    metrics = ["accuracy", "f1", "precision", "recall"]
    comparison_dfs = {}

    for metric in metrics:
        comparison_dfs[metric] = only_attacks[["name", metric]].rename(columns={metric: f"{metric}_model_1"}).merge(
            all_types[["name", metric]].rename(columns={metric: f"{metric}_model_2"}), on="name"
        ).merge(
            two_classes[["name", metric]].rename(columns={metric: f"{metric}_model_3"}), on="name"
        )

    # Visualisierung aller Metriken
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bar_width = 0.3
    x_labels = only_attacks["name"]
    x = range(len(x_labels))

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df_metric = comparison_dfs[metric]

        bars1 = ax.bar(x, df_metric[f"{metric}_model_1"], width=bar_width, label=f"{metric} - Model Only Attacks")
        bars2 = ax.bar([p + bar_width for p in x], df_metric[f"{metric}_model_2"], width=bar_width, label=f"{metric} - Model All Attack Types")
        bars3 = ax.bar([p + 2 * bar_width for p in x], df_metric[f"{metric}_model_3"], width=bar_width, label=f"{metric} - Model Normal & Klasse-X")

            # Werte über die Balken schreiben
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks([p + bar_width for p in x])
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Vergleich der {metric.capitalize()}-Werte zwischen den drei Modellen")
        ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(model_comparison_path, "comparison_overall_metrics.pdf"), format='pdf', dpi=1000)   
    plt.close()

"""
def plot_all_only_attacks_one_vs_all_metrics():
    # Definieren der Tabellen
    result_model_trained_only_on_attacks = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.569242, 0.615907, 0.890924, 0.897534, 0.976801],
        "f1": [0.0, 0.607034, 0.62521, 0.625365, 0.149593],
        "precision": [0.0, 0.458805, 0.538744, 0.51482, 0.110843],
        "recall": [0.0, 0.896755, 0.744735, 0.796365, 0.23]
    })

    result_model_trained_on_all_attack_types = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.880767, 0.935504, 0.874867, 0.959812, 0.98035],
        "f1": [0.87286, 0.896996, 0.358718, 0.814344, 0.196007],
        "precision": [0.807191, 0.950886, 0.479635, 0.808052, 0.153846],
        "recall": [0.95016, 0.848887, 0.286492, 0.820735, 0.27]
    })

    # Definieren der dritten Tabelle
    two_classes = pd.DataFrame({
        "name": ["normal", "DoS", "R2L", "Probe", "U2R"],
        "accuracy": [0.4997, 0.870742, 0.878948, 0.730882, 0.96305],
        "f1": [0.631293, 0.811733, 0.382719, 0.393967, 0.097508],
        "precision": [0.631293, 0.783292, 0.507499, 0.259816, 0.062241],
        "recall": [0.923489, 0.842317, 0.30719, 0.814539, 0.225]
    })

    # Zusammenführen der F1-Werte aller drei Tabellen
    comparison_df = result_model_trained_only_on_attacks[["name", "f1"]].rename(columns={"f1": "f1_model_1"}).merge(
        result_model_trained_on_all_attack_types[["name", "f1"]].rename(columns={"f1": "f1_model_2"}), on="name").merge(
            two_classes[["name", "f1"]].rename(columns={"f1": "f1_model_3"}), on="name")

    # Erstellen eines Balkendiagramms zum Vergleich der F1-Werte
    plt.figure(figsize=(12, 6))
    bar_width = 0.3
    x_labels = comparison_df["name"]
    x = range(len(x_labels))

    plt.bar(x, comparison_df["f1_model_1"], width=bar_width, label="F1 - Modell Only Attacks")
    plt.bar([p + bar_width for p in x], comparison_df["f1_model_2"], width=bar_width, label="F1 - Modell All Attack Types")
    plt.bar([p + 2 * bar_width for p in x], comparison_df["f1_model_3"], width=bar_width, label="F1 - Modell Normal & Klasse-X")

    plt.xticks([p + bar_width for p in x], x_labels)
    plt.ylabel("F1-Score")
    plt.title("Vergleich der F1-Scores zwischen \"All\", \"Attacks\" und \"Normal & Klasse-X\" Modellen")
    plt.legend()
    plt.savefig(os.path.join(model_comparison_path, "comparison_one_vs_all_metrics.pdf"), format='pdf', dpi=1000)
"""

def plot_roc_curve(fpr, tpr, roc_auc, path):
    # Plotten der ROC-Kurve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, 'roc_curve.pdf'), format='pdf', dpi=1000)
    plt.close()