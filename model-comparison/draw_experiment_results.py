import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import plot_accuracy_vs_hidden_layers, plot_accuracy_vs_learning_rate, plot_boxplot, plot_confusion_matrix_comparison, plot_heatmap_for_configs, plot_line_diagram, plot_line_diagram_to_compare_accuracy_of_various_models, plot_metrics_comparison_of_original_and_my_code, plot_one_vs_all_metrics_comparison_between_all_only_attacks_and_normal_and_class

cwd = os.getcwd()
line_diagram_path = os.path.join(cwd)

data = {
    'LR': [0.00025, 0.001, 0.01],
    'Original Code': [0.8156, 0.8234, 0.7987],
    'My Code': [0.8345, 0.8456, 0.8123]
}
df = pd.DataFrame(data)
plot_line_diagram(df)
plot_metrics_comparison_of_original_and_my_code()
plot_line_diagram_to_compare_accuracy_of_various_models()
plot_boxplot()
plot_heatmap_for_configs()
plot_accuracy_vs_hidden_layers()
plot_accuracy_vs_learning_rate()
plot_confusion_matrix_comparison()
plot_one_vs_all_metrics_comparison_between_all_only_attacks_and_normal_and_class()
#plot_all_only_attacks_one_vs_all_metrics()