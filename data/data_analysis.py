import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import NSL_KDD_FORMATTED_TEST_PATH, NSL_KDD_FORMATTED_TRAIN_PATH, ORIGINAL_KDD_TEST, ORIGINAL_KDD_TRAIN

from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from data.nsl_kdd_data_manager import NslKddDataManager
from utils.log_config import logger_setup

test_data = NslKddDataManager(
    trainset_path=ORIGINAL_KDD_TRAIN,
    testset_path=ORIGINAL_KDD_TEST,
    formated_trainset_path=NSL_KDD_FORMATTED_TRAIN_PATH,
    formated_testset_path=NSL_KDD_FORMATTED_TEST_PATH,
    dataset_type="test"
)

training_data = NslKddDataManager(
    trainset_path=ORIGINAL_KDD_TRAIN,
    testset_path=ORIGINAL_KDD_TEST,
    formated_trainset_path=NSL_KDD_FORMATTED_TRAIN_PATH,
    formated_testset_path=NSL_KDD_FORMATTED_TEST_PATH,
    dataset_type="train"
)



def view_dataset_statistics(data: NslKddDataManager):
    """
    View dataset statistics
    """
    # View dataset statistics
    print(data.df.describe()) # describe(include='all') for all columns
    print('Total number of rows: {0}\n\n'.format(data.get_shape()[0]))

def plot_correlation_matrix(data):
    """
    Plot correlation matrix
    """
    # Plot correlation matrix
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
    plt.close()

def visualize_relationships_within_data(data: NslKddDataManager):
    """
    Visualize relationships within data
    """
    test  = NslKddDataManager(dataset_type="test")
    normal_data, normal_attacks = test.update_samples_for_attack_type(["normal"], 0)
    normal_states, normal_labels = test.get_batch(batch_size=10)
    test  = NslKddDataManager(dataset_type="test")
    dos_data, dos_attacks = test.update_samples_for_attack_type(["DoS"], 0)
    test  = NslKddDataManager(dataset_type="test")
    probe_data, probe_attacks = test.update_samples_for_attack_type(["Probe"], 0)
    test  = NslKddDataManager(dataset_type="test")
    r2l_data, r2l_attacks = test.update_samples_for_attack_type(["R2L"], 0)
    test  = NslKddDataManager(dataset_type="test")
    u2r_data, u2r_attacks = test.update_samples_for_attack_type(["U2R"], 0)

    normal_data = normal_data[normal_data[normal_attacks]].sample(n=1, random_state=42)
    
    dos_data = dos_data[dos_data[dos_attacks]].sample(n=10, random_state=42)
    probe_data = probe_data[probe_data[probe_attacks]].sample(n=10, random_state=42)
    r2l_data = r2l_data[r2l_data[r2l_attacks].sample(n=10, random_state=42)]
    u2r_data = u2r_data[u2r_data[u2r_attacks].sample(n=10, random_state=42)]

    sns.pairplot(normal_states)
    plt.show()
    plt.savefig("pairplot_normal.pdf", format="pdf")
    plt.close()
    sns.pairplot(dos_data)
    plt.show()
    plt.savefig("pairplot_dos.pdf", format="pdf")
    plt.close()
    sns.pairplot(probe_data)
    plt.show()
    plt.savefig("pairplot_probe.pdf", format="pdf")
    plt.close()
    sns.pairplot(r2l_data)
    plt.show()
    plt.savefig("pairplot_r2l.pdf", format="pdf")
    plt.close()
    data = pd.concat([normal_data, dos_data, probe_data, r2l_data, u2r_data])
    sns.pairplot(data)
    plt.show()
    plt.savefig("pairplot.pdf", format="pdf")
    plt.close()
    

    #@title Code - View pairplot
    #sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])

def balance_r2l_and_u2r_data():
    """
    Balance R2L and U2R data and save the updated DataFrames to new CSV files
    """
    test_data = NslKddDataManager(
        trainset_path=ORIGINAL_KDD_TRAIN,
        testset_path=ORIGINAL_KDD_TEST,
        formated_trainset_path=NSL_KDD_FORMATTED_TRAIN_PATH,
        formated_testset_path=NSL_KDD_FORMATTED_TEST_PATH,
        dataset_type="test")

    training_data = NslKddDataManager(
        trainset_path=ORIGINAL_KDD_TRAIN,
        testset_path=ORIGINAL_KDD_TEST,
        formated_trainset_path=NSL_KDD_FORMATTED_TRAIN_PATH,
        formated_testset_path=NSL_KDD_FORMATTED_TEST_PATH,
        dataset_type="train")
    
    # Balance R2L data
    # Shuffle and extract 725 samples from warezmaster and 950 samples from guess_passwd
    guess_passwd = test_data.df[test_data.df["guess_passwd"] == 1]
    guess_passwd = guess_passwd.sample(frac=1)
    guess_passwd_data_to_move = guess_passwd[:950]
    guess_passwd = guess_passwd[950:]
    
    warezmaster = test_data.df[test_data.df["warezmaster"] == 1]
    warezmaster = warezmaster.sample(frac=1)
    warezmaster_data_to_move = warezmaster[:725]
    warezmaster = warezmaster[725:]

    # Remove the extracted samples from the test data
    test_data.df = test_data.df[~test_data.df.index.isin(guess_passwd_data_to_move.index)]
    test_data.df = test_data.df[~test_data.df.index.isin(warezmaster_data_to_move.index)]


    # Balance U2R data
    httptunnel = test_data.df[test_data.df["httptunnel"] == 1]
    httptunnel = httptunnel.sample(frac=1)
    httptunnel_data_to_move = httptunnel[:118]
    httptunnel = httptunnel[118:]

    rootkit = test_data.df[test_data.df["rootkit"] == 1]
    rootkit = rootkit.sample(frac=1)
    rootkit_data_to_move = rootkit[:10]
    rootkit = rootkit[10:]

    buffer_overflow = training_data.df[training_data.df["buffer_overflow"] == 1]
    buffer_overflow_data_to_move = buffer_overflow[:]

    # Remove the extracted samples from the test data
    test_data.df = test_data.df[~test_data.df.index.isin(httptunnel_data_to_move.index)]
    test_data.df = test_data.df[~test_data.df.index.isin(rootkit_data_to_move.index)]
    training_data.df = training_data.df[~training_data.df.index.isin(buffer_overflow_data_to_move.index)]

    # Move the extracted samples to the training data
    test_data.df = pd.concat([test_data.df, buffer_overflow_data_to_move]) # U2R samples
    training_data.df = pd.concat([training_data.df, httptunnel_data_to_move]) # U2R samples
    training_data.df = pd.concat([training_data.df, rootkit_data_to_move])
    training_data.df = pd.concat([training_data.df, guess_passwd_data_to_move])
    training_data.df = pd.concat([training_data.df, warezmaster_data_to_move])
    
    # Save the updated DataFrames to new CSV files
    test_data.df.to_csv("data/datasets/formated/balanced_test_data.csv", index=False)
    training_data.df.to_csv("data/datasets/formated/balanced_training_data.csv", index=False)
    

# TODO: Add Precision-recall curve (for imbalanced data) and ROC curve
if __name__ == "__main__":
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger_setup(timestamp_begin, "-data-analysis")
    #view_dataset_statistics(test_data)
    #view_dataset_statistics(training_data)
    #visualize_relationships_within_data(training_data.df)
    #visualize_relationships_within_data(test_data)
    #plot_correlation_matrix(test_data.df)
    #plot_correlation_matrix(training_data.df)
    balance_r2l_and_u2r_data()



