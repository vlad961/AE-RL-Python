import io
import logging
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import shutil
import tensorflow as tf

from datetime import datetime
from matplotlib import pyplot as plt
from models.agent import Agent
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
##################
# Helper Methods #
##################
cwd = os.getcwd()
model_comparison_path = os.path.join(cwd, "model-comparison")

def download_file(url:str, local_filename:str):
    """
    Download a file from a given URL and save it locally.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The local path where the file will be saved.

    Returns:
        str: The local path where the file was saved.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def save_model(agent: Agent, model_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    # Get the model summary and save it to a file
    agent.model_network.model.save(model_path)
    logging.info(f"Model '{agent.model_network.model_name}' saved in: {model_path}")
    logging.info(f"Model summary:\n{get_model_summary(agent)}")

def get_model_summary(model):
    stream = io.StringIO()
    if isinstance(model, Agent):
        model.model_network.model.summary(print_fn=lambda x: stream.write(x + "\n"))
    else:
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    return summary_str

def download_datasets_if_missing(kdd_train:str, kdd_test:str):
    # If the data files for some reason do not exist, download them from the repo this work is based on.
    if (not os.path.exists(kdd_train)):
        kdd_train_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTrain%2B.txt"
        download_file(kdd_train_url, kdd_train)
        logging.info("Downloaded: {}\nSaved in: {}", kdd_train_url, kdd_train)
    if (not os.path.exists(kdd_test)):
        kdd_test_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTest%2B.txt"
        download_file(kdd_test_url, kdd_test)
        logging.info("Downloaded: {}\nSaved in: {}", kdd_test_url, kdd_test)

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

def print_total_runtime(script_start_time):
    total_runtime = datetime.now() - script_start_time
    # Convert total runtime to hours, minutes, and seconds
    total_seconds = int(total_runtime.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Total runtime: {hours:02}:{minutes:02}:{seconds:02}")
    logging.info(f"End of the script at: {datetime.now()}")



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
    plt.savefig(os.path.join(path, 'train_reward_loss.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(path, 'train_reward_loss.pdf'), format='pdf', dpi=1000)
    plt.close()

    logging.info(f"Plotted rewards and losses during training in: {path}")
    return plt

def plot_attack_distributions(attacks_by_epoch, attack_names, attack_labels_list, path) -> plt.Figure:
    bins=np.arange(len(attack_names))
    # Plot attacks distribution alongside
    plt.figure(2,figsize=[12,12])
    plt.xticks([])
    plt.yticks([])
    plt.title("Attacks distribution throughout episodes")
    for indx,e in enumerate([0,10,20,30,40,60,70,80,90]):
    #for indx,e in enumerate([0, 1, 2]):
        plt.subplot(3,3,indx+1)
        plt.hist(attacks_by_epoch[e], bins=bins, width=0.9, align='left')
        plt.xlabel("{} epoch".format(e))
        plt.xticks(bins, attack_names, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'attacks_distribution.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(path, 'attacks_distribution.pdf'), format='pdf', dpi=1000)
    plt.close()

    # Plot attacks distribution alongside
    plt.figure(3,figsize=[10,10])
    plt.xticks([])
    plt.yticks([])
    plt.title("Attacks (mapped) distribution throughout  episodes")
    for indx,e in enumerate([0,10,20,30,40,60,70,80,90]):
    #for indx,e in enumerate([0, 1, 2]):
        plt.subplot(3,3,indx+1)
        plt.bar(range(5), attack_labels_list[e], tick_label = ['Normal','Dos','Probe','R2L','U2R'])
        plt.xlabel("{} epoch".format(e))

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'attacks_mapped_distribution.eps'), format='eps', dpi=1000)
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
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(path, 'confusion_matrix.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(path, 'confusion_matrix.pdf'), format='pdf', dpi=1000)
    plt.close()

    logging.info(f"Plotted confusion matrix in: {path}")
    #fig, ax = plt.subplots(figsize=(8, 6))  
    #im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)

    #ax.title(title)
    #fig.colorbar(im)
    #tick_marks = np.arange(len(classes))
    #ax.set_xticks(tick_marks)
    #ax.set_yticks(tick_marks)
    #ax.set_xticklabels(classes, rotation=45)
    #ax.set_yticklabels(classes)
    """
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(path, 'confusion_matrix.eps'), format='eps', dpi=1000)
    plt.close()
    """

    return plt


#####################
# Metrics functions #
#####################
def get_cf_matrix(true_labels, predicted_labels):
    cnf_matrix = confusion_matrix(true_labels, predicted_labels)
    return cnf_matrix

def calculate_f1_scores_per_class(predicted_actions, attack_types, true_labels):
    predicted_actions_dummies = pd.get_dummies(predicted_actions)
    posible_actions = np.arange(len(attack_types))
    for non_existing_action in posible_actions:
        if non_existing_action not in predicted_actions_dummies.columns:
            predicted_actions_dummies[non_existing_action] = np.uint8(0)
    true_labels_dummies = pd.get_dummies(true_labels)

    normal_f1_score = f1_score(true_labels_dummies[0].values, predicted_actions_dummies[0].values)
    dos_f1_score = f1_score(true_labels_dummies[1].values, predicted_actions_dummies[1].values)
    probe_f1_score = f1_score(true_labels_dummies[2].values, predicted_actions_dummies[2].values)
    r2l_f1_score = f1_score(true_labels_dummies[3].values, predicted_actions_dummies[3].values)
    u2r_f1_score = f1_score(true_labels_dummies[4].values, predicted_actions_dummies[4].values)
    overall_f1_score = f1_score(true_labels, predicted_actions, average='weighted')

    return [normal_f1_score, dos_f1_score, probe_f1_score, r2l_f1_score, u2r_f1_score, overall_f1_score]

def print_aggregated_performance_measures(predicted_actions, true_labels):
    logging.info('Performance measures on Test data')
    logging.info('Accuracy =  {:.4f}'.format(accuracy_score(true_labels, predicted_actions)))
    logging.info('F1 =  {:.4f}'.format(f1_score(true_labels, predicted_actions, average='weighted')))
    logging.info('Precision_score =  {:.4f}'.format(precision_score(true_labels, predicted_actions, average='weighted')))
    logging.info('recall_score =  {:.4f}'.format(recall_score(true_labels, predicted_actions, average='weighted')))

def calculate_one_vs_all_metrics(true_attack_type_indices, actions):
    mapa = {0:'normal', 1:'DoS', 2:'Probe',3:'R2L',4:'U2R'}
    yt_app = pd.Series(true_attack_type_indices).map(mapa)

    perf_per_class = pd.DataFrame(index=range(len(yt_app.unique())),columns=['name', 'accuracy','f1', 'precision','recall'])
    for i,x in enumerate(pd.Series(yt_app).value_counts().index):
        y_test_hat_check = pd.Series(actions).map(mapa).copy()
        y_test_hat_check[y_test_hat_check != x] = 'OTHER'
        yt_app = pd.Series(true_attack_type_indices).map(mapa).copy()
        yt_app[yt_app != x] = 'OTHER'
        ac=accuracy_score(yt_app, y_test_hat_check)
        f1=f1_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
        pr=precision_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
        re=recall_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
        perf_per_class.iloc[i]=[x,ac,f1,pr,re]
        
    return perf_per_class

def calculate_general_overview_per_attack_type(attack_types, estimated_labels, estimated_correct_labels, true_labels, f1_scores, mismatch) -> pd.DataFrame:
    outputs_df = pd.DataFrame(index = attack_types, columns = ["Estimated", "Correct", "Total", "F1_score", "Mismatch"])
    for indx, _ in enumerate(attack_types):
        outputs_df.iloc[indx].Estimated = estimated_labels[indx]
        outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
        outputs_df.iloc[indx].Total = true_labels[indx]
        outputs_df.iloc[indx].F1_score = f1_scores[indx]*100
        outputs_df.iloc[indx].Mismatch = abs(mismatch[indx])

    # Add a row for the general F1 score
    general_f1_score = f1_scores[-1]
    general_row = pd.DataFrame([{
        "Estimated": "",
        "Correct": "",
        "Total": "",
        "F1_score": general_f1_score * 100,
        "Mismatch": ""
    }], index=["General"])

    outputs_df = pd.concat([outputs_df, general_row])

    return outputs_df

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
    plt.savefig(os.path.join(path, 'roc_curve.pdf'), format='pdf', dpi=1000)
    plt.close()