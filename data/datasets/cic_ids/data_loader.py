from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, matthews_corrcoef, average_precision_score, roc_auc_score, precision_recall_curve, auc, roc_curve
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np

def load_data(data_file_path, verbose=True, train_size=50000, test_size=None, downsample_benign=False, benign_sample_size=250000):
    if ".csv" in data_file_path:
        df = pd.read_csv(data_file_path)
    elif ".feather" in data_file_path:
        df = pd.read_feather(data_file_path)
    elif ".parquet" in data_file_path:
        df = pd.read_parquet(data_file_path)
    else:
        raise ValueError("File type not supported. Please provide a .csv, .feather, or .parquet file.")
    
    if downsample_benign:
        if "Label" not in df.columns:
            raise ValueError("Column 'Label' not found in data.")
        benign_df = df[df["Label"] == "Benign"]
        if benign_df.shape[0] < benign_sample_size:
            raise ValueError(f"Requested {benign_sample_size} benign samples, but only {benign_df.shape[0]} available.")
        df = benign_df.sample(n=benign_sample_size, random_state=42)


    labels = df["Label"].copy()  # keep full label info for mapping
    Y = df["Label"].map(lambda x: 1 if (x == "Benign") else -1)
    df.drop(columns=["Label", "Timestamp", "Destination Port"], inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(
        df, Y, train_size=train_size, test_size=test_size,
        shuffle=True, stratify=Y, random_state=42)
    
    if verbose:
        print("***** Train Data *****")
        print(labels.loc[y_train.index].value_counts())
        print("***** Test Data *****")
        print(labels.loc[y_test.index].value_counts())
    return x_train, x_test, y_train, y_test, labels.loc[y_train.index], labels.loc[y_test.index]

def load_data_fraud(data_file_path, verbose=True):
    if ".csv" in data_file_path:
        df = pd.read_csv(data_file_path)
    elif ".feather" in data_file_path:
        df = pd.read_feather(data_file_path)
    elif ".parquet" in data_file_path:
        df = pd.read_parquet(data_file_path)
    else:
        raise ValueError("File type not supported. Please provide a .csv, .feather, or .parquet file.")
    
    labels = df["Label"].copy()  # keep full label info for mapping
    Y = df["Label"].map(lambda x: 1 if (x == "Benign") else -1)
    df.drop(columns=["Label", "Timestamp", "Destination Port"], inplace=True)

    if verbose:
        print("***** Data *****")
        print(labels.value_counts())
        print("Total number of samples:", df.shape[0])
    return df, Y, labels

# def tune_hyperparam(X, y, pipe, hyper_param, folds=5, verbose=1, n_jobs=-1, refit="f1", scoring=None):
#     if scoring is None:
#         scoring = {
#             "accuracy": "accuracy",
#             "recall": make_scorer(recall_score, pos_label=-1, zero_division=0),
#             "precision": make_scorer(precision_score, pos_label=-1, zero_division=0),
#             "f1": make_scorer(f1_score, pos_label=-1, zero_division=0),
#             "mcc": make_scorer(matthews_corrcoef),
#             "auroc": make_scorer(roc_auc_score, needs_threshold=True),
#             "average_precision": make_scorer(average_precision_score, needs_threshold=True)
#         }
#     cv = GridSearchCV(pipe, hyper_param, scoring=scoring, cv=folds, refit=refit, return_train_score=False, n_jobs=n_jobs, verbose=verbose)
#     cv.fit(X, y)
#     if refit:
#         print(f"Best parameters set found on development set: {cv.best_params_}")
#         print(f"Mean score of best model ({refit}): {cv.best_score_}")
#     return cv

def anomaly_scores(original, transformed):
    sse = np.sum((original - transformed)**2, axis=1)
    return sse

def evaluate_results(y_true, score):
    precision, recall, threshold = precision_recall_curve(y_true, score, pos_label=-1)
    au_precision_recall = auc(recall, precision)
    results = pd.DataFrame({'precision': precision, 'recall': recall})
    results["f1"] = 2*precision*recall/(precision+recall)
    max_index = results["f1"].idxmax()
    best = results.loc[results["f1"].idxmax()]
    best["threshold"] = threshold[max_index]
    best["au_precision_recall"] = au_precision_recall
    fpr, tpr, thresholds = roc_curve(y_true, score, pos_label=-1)
    best["auroc"] = auc(fpr, tpr)
    return best

def evaluate_predictions(y_true, y_pred):
    results = {}
    results['recall'] = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    results['precision'] = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    results['f1'] = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
    return results

def evaluate_test_data(y_true, score, threshold):
    y_pred = np.array([1 if score < threshold else -1 for score in score])
    results = evaluate_predictions(y_true, y_pred)
    precision, recall, threshold = precision_recall_curve(y_true, score, pos_label=-1)
    results['au_precision_recall'] = auc(recall, precision)
    fpr, tpr, thresholds = roc_curve(y_true, score, pos_label=-1)
    results["auroc"] = auc(fpr, tpr)
    return results