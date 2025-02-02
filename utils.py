
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import time


from data.data_cls import DataCls
from models.helpers import get_model_summary

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
    x_ticks = np.arange(len(def_reward_chain))
    x_labels = [f'{i+1}' for i in range(len(def_reward_chain))]

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

    return plt

def plot_attack_distributions(attacks_by_epoch, attack_names, attack_labels_list, path) -> plt.Figure:
    bins=np.arange(23)
    # Plot attacks distribution alongside
    plt.figure(2,figsize=[12,5])
    plt.xticks([])
    plt.yticks([])
    plt.title("Attacks distribution throughout episodes")
    for indx,e in enumerate([0,70,90]):
    #for indx,e in enumerate([0]):
        plt.subplot(1,3,indx+1)
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
    #for indx,e in enumerate([0]):
        plt.subplot(3,3,indx+1)
        plt.bar(range(5), attack_labels_list[e], tick_label = ['Normal','Dos','Probe','R2L','U2R'])
        plt.xlabel("{} epoch".format(e))

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'attacks_mapped_distribution.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(path, 'attacks_mapped_distribution.pdf'), format='pdf', dpi=1000)
    plt.close()

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
    sn.heatmap(cm, annot=True, fmt='g', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(path, 'confusion_matrix.eps'), format='eps', dpi=1000)
    plt.savefig(os.path.join(path, 'confusion_matrix.pdf'), format='pdf', dpi=1000)
    plt.close()
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

def test_trained_agent_quality(path_to_model, plots_path):
    model = tf.keras.models.load_model(path_to_model)
    logging.info(f"Model '{model.name}' loaded from: {path_to_model}")
    logging.info(f"Model summary:\n{get_model_summary(model)}")

    # Define environment, game, make sure the batch_size is the same in train
    test_data = DataCls(dataset_type='test')

    total_reward = 0
    true_labels = np.zeros(len(test_data.attack_types),dtype=int)
    predicted_labels = np.zeros(len(test_data.attack_types),dtype=int)
    predicted_correct_labels = np.zeros(len(test_data.attack_types),dtype=int)

    #states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
    states, labels = test_data.get_full() # get test data and true labels.

    start_time=time.time()
    q = model.predict(states)
    actions = np.argmax(q,axis=1) # get the action with the highest Q-value -> the predicted attack type

    true_attack_type_indices=[] # list of true attack types as indices (0-4). 0=normal, 1=dos, 2=probe, 3=r2l, 4=u2r. Length = number of samples
    for _, label in labels.iterrows():
        label_class = label.idxmax()
        attack_type = test_data.attack_map[label_class]
        attack_type_index = test_data.attack_types.index(attack_type)
        true_attack_type_indices.append(attack_type_index)

    labels_per_attack_type, counts = np.unique(true_attack_type_indices, return_counts=True)
    true_labels[labels_per_attack_type] += counts

    for indx, a in enumerate(actions):
        predicted_labels[a] +=1
        if a == true_attack_type_indices[indx]:
            total_reward += 1
            predicted_correct_labels[a] += 1

    f1_scores = calculate_unique_f1_scores_per_class(actions, test_data.attack_types, true_attack_type_indices)
    mismatch = predicted_labels - true_labels
    acc = float(100 * total_reward / len(states))

    logging.info(f"Overall overview\nTotal reward: {total_reward} | Number of samples: {len(states)} | Accuracy = {acc:.2f}%")
    
    outputs_df = calculate_general_overview_per_attack_type(test_data.attack_types, predicted_labels, predicted_correct_labels, true_labels, f1_scores, mismatch)
    logging.info(f"Overall overview per attack type\n{outputs_df}")

    aggregated_data_test = np.array(true_attack_type_indices)
    print_aggregated_performance_measures(actions, aggregated_data_test)

    cnf_matrix = confusion_matrix(aggregated_data_test, actions)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=test_data.attack_types, path=plots_path, normalize=True,
                        title='Normalized confusion matrix')

    perf_per_class = calculate_one_vs_all_metrics(true_attack_type_indices, actions)
    logging.info("\r\nOne vs All metrics: \r\n{}".format(perf_per_class))
    result = model.evaluate(states, pd.get_dummies(true_attack_type_indices), verbose=2)
    loss, acc_model, precision, recall = result
    logging.info(f"Model metrics: loss={loss}, accuracy={acc_model}, precision={precision}, recall={recall}")
    logging.info(f"Time needed for testing: {time.time()-start_time}")

def calculate_unique_f1_scores_per_class(predicted_actions, attack_types, true_labels):
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

    return outputs_df