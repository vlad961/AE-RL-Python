
from datetime import datetime
import logging
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from data.data_cls import DataCls
from models.helpers import logger_setup

test_data  = DataCls(dataset_type="test")
training_data = DataCls(dataset_type="train")



def view_dataset_statistics(data: DataCls):
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

def visualize_relationships_within_data(data: DataCls):
    """
    Visualize relationships within data
    """
    test  = DataCls(dataset_type="test")
    normal_data, normal_attacks = test.get_samples_for_attack_type(["normal"], 0)
    normal_states, normal_labels = test.get_batch(batch_size=10)
    test  = DataCls(dataset_type="test")
    dos_data, dos_attacks = test.get_samples_for_attack_type(["DoS"], 0)
    test  = DataCls(dataset_type="test")
    probe_data, probe_attacks = test.get_samples_for_attack_type(["Probe"], 0)
    test  = DataCls(dataset_type="test")
    r2l_data, r2l_attacks = test.get_samples_for_attack_type(["R2L"], 0)
    test  = DataCls(dataset_type="test")
    u2r_data, u2r_attacks = test.get_samples_for_attack_type(["U2R"], 0)

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


# TODO: Add Precision-recall curve (for imbalanced data) and ROC curve
if __name__ == "__main__":
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger_setup(timestamp_begin, "-data-analysis")
    view_dataset_statistics(test_data)
    #view_dataset_statistics(training_data)
    #visualize_relationships_within_data(training_data.df)
    visualize_relationships_within_data(test_data)
    plot_correlation_matrix(test_data.df)
    #plot_correlation_matrix(training_data.df)