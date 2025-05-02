import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def filter_common_decisions(file_paths):
    """
    Filters rows from multiple CSV files based on common values
    in the 'decision' column ('confirmed', 'rejected', 'tentative')
    and returns a dictionary for each decision type.

    Parameters:
    - file_paths: List of paths to the CSV files.

    Returns:
    - A dictionary with keys: 'confirmed', 'rejected', 'tentative',
      and values: list of feature names (first-column values) that meet the conditions.
    """
    if len(file_paths) < 2:
        raise ValueError("At least two CSV files are required for comparison.")

    # Read all CSV files into DataFrames
    dataframes = [pd.read_csv(file) for file in file_paths]

    # Initialize a dictionary to store common features for each decision
    result = {"Confirmed": [], "Rejected": [], "Tentative": []}

    # Loop over the decisions to filter each one individually
    for decision in ["Confirmed", "Rejected", "Tentative"]:
        # Start with filtering rows for the current decision in the first file
        common_filtered = dataframes[0][dataframes[0]["decision"] == decision]

        # Only keep the first column (feature names) and 'decision'
        common_filtered = common_filtered.iloc[:, [0, common_filtered.columns.get_loc("decision")]]


        # Loop through remaining files and find rows with matching 'decision' column
        for df in dataframes[1:]:
            filtered = df[df["decision"] == decision]
            filtered = filtered.iloc[:, [0, filtered.columns.get_loc("decision")]]  # Keep first column and 'decision'

            # Merge with common_filtered on the first column (assuming first column is feature names)
            common_filtered = pd.merge(
                common_filtered,
                filtered,
                on=common_filtered.columns[0],
                how="inner",
                validate="one_to_one",
                suffixes=(None, "_dup") # Resolves suffix conflicts
            )

        # Add the common features (first column values) to the dictionary
        result[decision] = list(common_filtered.iloc[:, 0])

    return result

def visualize_feature_relevance(file_paths, labels=None, decision_filter="Confirmed"):
    """
    Visualize stable features over different subsets for a specific decision type.

    Parameters:
    - file_paths: List of paths to the CSV files.
    - labels: Optional list of labels for each subset.
    - decision_filter: The decision type to filter before visualizing ('Confirmed', 'Rejected', 'Tentative').

    Returns:
    - A heatmap that shows the stable features filtered as the specified decision type over different subsets.
    """
    if labels and len(labels) != len(file_paths):
        raise ValueError("The amount of labels must match the amount of file paths.")

    if labels is None:  # In case no labels are given, set generic labels
        labels = [f"Dataset {i + 1}" for i in range(len(file_paths))]

    feature_relevance = pd.DataFrame()

    # Iterate files and extract features for the specified decision type
    for file_path, label in zip(file_paths, labels):
        df = pd.read_csv(file_path)

        # Filter only features with the specified decision type
        selected_features = df[df["decision"] == decision_filter].iloc[:, 0].values

        relevance = pd.Series(1, index=selected_features, name=label)
        feature_relevance = pd.concat([feature_relevance, relevance], axis=1)

    # Set features that are not in the specified decision type to 0.
    feature_relevance = feature_relevance.fillna(0)

    if feature_relevance.empty:
        print(f"WARNING: No relevant features found for decision '{decision_filter}'.")
        return

    # Create the heatmap
    plt.figure(figsize=(10, len(feature_relevance) // 2))

    # Create discrete color map for 0 and 1
    colors = ["#f0f0f0", "#1f78b4"]
    cmap = ListedColormap(colors)  # Define the colormap manually

    # Set custom boundaries for a discrete legend
    bounds = np.array([0, 0.5, 1])  # Boundaries for the color levels
    norm = BoundaryNorm(bounds, ncolors=len(colors))  # Normalize to boundaries

    sns.heatmap(
        feature_relevance,
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={
            "ticks": [0, 1],  # Only show ticks for 0 and 1
            "label": f"Feature Relevance (0: Not {decision_filter}, 1: {decision_filter})",
        },
        norm=norm,
    )

    plt.title(
        f"Representation of stable features identified as '{decision_filter}'\n"
        "across different subsets using Boruta analyses."
    )
    plt.xlabel("Subsets")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(f"boruta_feature_relevance_{decision_filter.lower()}.pdf", format="pdf")




boruta_results = ["/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta/rand-vs-strat-new/rand_10_percent/01_boruta_results_before_tentative_decision.csv",
                  "/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta/rand-vs-strat-new/strat_10_percent/01_boruta_results_before_tentative_decision.csv",
                  "/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta/strat-vs-balanced-new/balanced/01_boruta_results_before_tentative_decision.csv",
                  "/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta/strat-manually-filtered-highly-correlated-data/01_boruta_results_before_tentative_decision.csv",
#                  "/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta/single-attack-type-analysis/normal-vs-dos/01_boruta_result_DoS_vs_Normal_before_tentative.csv",
#                  "/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta/single-attack-type-analysis/normal-vs-probe/01_boruta_result_Probe_vs_Normal_before_tentative.csv",
#                  "/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta/single-attack-type-analysis/normal-vs-r2l/01_boruta_result_R2L_vs_Normal_before_tentative.csv",
#                  "/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta/single-attack-type-analysis/normal-vs-u2r/01_boruta_result_U2R_vs_Normal_before_tentative.csv"
          ]

boruta_subsets = ["Random", "Stratified", "Balanced", "Manually filtered highly correlated features"]
#boruta_subsets = ["Random", "Stratified", "Balanced", "Manually filtered highly correlated features", "Normal vs DoS", "Normal vs Probe", "Normal vs R2L", "Normal vs U2R"]

matching_features = filter_common_decisions(boruta_results)
matching_features.update({
    "confirmed_length": len(matching_features["Confirmed"]),
    "rejected_length": len(matching_features["Rejected"]),
    "tentative_length": len(matching_features["Tentative"]),
    "used_subsets": boruta_subsets,
    "file_paths": boruta_results,
})

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "boruta_subsets_feature_overlap_analysis_wo_single_types")
if not os.path.exists(path):
    os.makedirs(path)
path = os.path.join(path, "wo_single_type_analysis_features_overlapping_analysis.json")

with open(path, "w") as json_file:
    json.dump(matching_features, json_file, indent=4)
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "boruta_all_subsets_feature_overlap_analysis")
if not os.path.exists(path):
    os.makedirs(path)

path = os.path.join(path, "with_single_type_analysis_features_overlapping_analysis.json")
#with open(path, "w") as json_file:
#    json.dump(matching_features, json_file, indent=4)

visualize_feature_relevance(boruta_results, boruta_subsets, decision_filter="Confirmed")
visualize_feature_relevance(boruta_results, boruta_subsets, decision_filter="Rejected")
visualize_feature_relevance(boruta_results, boruta_subsets, decision_filter="Tentative")
print(matching_features)
print("bluah")