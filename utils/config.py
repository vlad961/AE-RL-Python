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
