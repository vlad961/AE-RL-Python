import numpy as np
import os

GLOBAL_SEED = 42
GLOBAL_RNG = np.random.default_rng(GLOBAL_SEED)

CWD = os.getcwd()
DATA_ROOT_DIR = os.path.join(CWD, "data/datasets/")
NSL_KDD_ORIGINAL_DIR = os.path.join(DATA_ROOT_DIR, "origin-kaggle-com/nsl-kdd/")
DATA_FORMATTED_DIR = os.path.join(DATA_ROOT_DIR, "formated/")
NSL_KDD_FORMATTED_TRAIN_PATH = os.path.join(DATA_FORMATTED_DIR, "formated_training_data.csv") # formated_train_adv balanced_training_data
NSL_KDD_FORMATTED_TEST_PATH = os.path.join(DATA_FORMATTED_DIR, "formated_test_data.csv") # formated_test_adv balanced_test_data
ORIGINAL_KDD_TRAIN = os.path.join(NSL_KDD_ORIGINAL_DIR, "KDDTrain+.txt")
ORIGINAL_KDD_TEST = os.path.join(NSL_KDD_ORIGINAL_DIR, "KDDTest+.txt")
TRAINED_MODELS_DIR = os.path.join(CWD, "models/trained-models/")
TEMPORARY_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "tmp_model.keras")


# CICIDS 2017
CICIDS_2017_DIR = os.path.join(DATA_ROOT_DIR, "cic_ids/cic-ids-2017/")
CICIDS_2017_CLEAN_DIR = os.path.join(CICIDS_2017_DIR, "clean/")
CICIDS_2017_CLEAN_ALL_DATA = os.path.join(CICIDS_2017_CLEAN_DIR, "all_data.feather")
CICIDS_2017_CLEAN_ALL_MALICIOUS = os.path.join(CICIDS_2017_CLEAN_DIR, "all_malicious.feather")
CICIDS_2017_CLEAN_ALL_BENIGN = os.path.join(CICIDS_2017_CLEAN_DIR, "all_benign.feather")
CICIDS_2017_CLEAN_ALL_DATA_CSV = os.path.join(CICIDS_2017_CLEAN_DIR, "all_data.csv")
CICIDS_2017_CLEAN_ALL_MALICIOUS_CSV = os.path.join(CICIDS_2017_CLEAN_DIR, "all_malicious.csv")
CICIDS_2017_CLEAN_ALL_BENIGN_CSV = os.path.join(CICIDS_2017_CLEAN_DIR, "all_benign.csv")
# CICIDS 2018
CICIDS_2018_DIR = os.path.join(DATA_ROOT_DIR, "cic_ids/cse-cic-ids-2018/")
CICIDS_2018_CLEAN_DIR = os.path.join(CICIDS_2018_DIR, "clean/")
CICIDS_2018_CLEAN_ALL_DATA = os.path.join(CICIDS_2018_CLEAN_DIR, "all_data.feather")
CICIDS_2018_CLEAN_ALL_MALICIOUS = os.path.join(CICIDS_2018_CLEAN_DIR, "all_malicious.feather")
CICIDS_2018_CLEAN_ALL_BENIGN = os.path.join(CICIDS_2018_CLEAN_DIR, "all_benign.feather")
CICIDS_2018_CLEAN_ALL_DATA_CSV = os.path.join(CICIDS_2018_CLEAN_DIR, "all_data.csv")
CICIDS_2018_CLEAN_ALL_MALICIOUS_CSV = os.path.join(CICIDS_2018_CLEAN_DIR, "all_malicious.csv")
CICIDS_2018_CLEAN_ALL_BENIGN_CSV = os.path.join(CICIDS_2018_CLEAN_DIR, "all_benign.csv")
