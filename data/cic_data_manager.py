import logging
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data.datasets.cic_ids.data_loader import load_data, load_data_fraud
from typing import List, Tuple
from utils.config import GLOBAL_RNG

ATTACK_TYPE_DDOS = "(D)DOS"
ATTACK_TYPE_BRUTE_FORCE = "Brute Force"
ATTACK_TYPE_WEB_ATTACK = "Web Attack"

cic_attack_map = {
    'Benign': 'Benign',
    'Bot': 'Botnet',
    'DDOS attack-LOIC-UDP': ATTACK_TYPE_DDOS,
    'DDoS': ATTACK_TYPE_DDOS,
    'DDoS attacks-LOIC-HTTP': ATTACK_TYPE_DDOS,
    'DDOS attack-HOIC': ATTACK_TYPE_DDOS,
    'DoS attacks-Hulk': ATTACK_TYPE_DDOS,
    'DoS attacks-GoldenEye': ATTACK_TYPE_DDOS,
    'DoS attacks-Slowloris': ATTACK_TYPE_DDOS,
    'DoS attacks-SlowHTTPTest': ATTACK_TYPE_DDOS,
    'DoS Hulk': ATTACK_TYPE_DDOS,
    'DoS GoldenEye': ATTACK_TYPE_DDOS,
    'DoS slowloris': ATTACK_TYPE_DDOS,
    'DoS Slowhttptest': ATTACK_TYPE_DDOS,
    'PortScan': 'Probe',
    'FTP-Patator': ATTACK_TYPE_BRUTE_FORCE,
    'SSH-Patator': ATTACK_TYPE_BRUTE_FORCE,
    'FTP-BruteForce': ATTACK_TYPE_BRUTE_FORCE,
    'SSH-Bruteforce': ATTACK_TYPE_BRUTE_FORCE,
    'Web Attack  Brute Force': ATTACK_TYPE_WEB_ATTACK,
    'Web Attack  XSS': ATTACK_TYPE_WEB_ATTACK,
    'Web Attack  Sql Injection': ATTACK_TYPE_WEB_ATTACK,
    'Brute Force -Web': ATTACK_TYPE_WEB_ATTACK,
    'Brute Force -XSS': ATTACK_TYPE_WEB_ATTACK,
    'SQL Injection': ATTACK_TYPE_WEB_ATTACK,
    'Infilteration': 'Infiltration', #CICIDS2018
    'Infiltration': 'Infiltration', #CICIDS2017
    'Heartbleed': 'Heartbleed',
}

@staticmethod
def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return label
    # Replace special characters and unify whitespace
    label = (
        label.replace('\x96', ' ')
             .replace('–', ' ')
             .replace('-', ' ')

    )
    # Fasse mehrere Leerzeichen zusammen zu genau einem
    label = re.sub(r'\s+', ' ', label)
    return label.strip()

# Normalisiere die Keys des Mappings
cic_attack_map = {normalize_label(k): v for k, v in cic_attack_map.items()}
cic_attack_map_one_vs_all = {k: ('Benign' if v == 'Benign' else 'attack') for k, v in cic_attack_map.items()}

class CICDataManager:
    def __init__(self, benign_path: str, malicious_path: str, is_cic_2018: bool, normalization: str = 'linear', one_vs_all: bool = False, target_attack_type: str = ATTACK_TYPE_DDOS, inter_dataset_run: bool = False, inter_dataset_benign_path: str = None, inter_dataset_malicious_path: str = None):
        self.benign_path = benign_path
        self.malicious_path = malicious_path
        self.is_cic_2018_training_set = is_cic_2018
        self.normalization = normalization
        self.one_vs_all = one_vs_all
        self.loaded = False
        self.df = None
        if self.one_vs_all:
            splits = self.load_one_vs_all_split(benign_path=benign_path,
                malicious_path=malicious_path,
                target_attack_type=target_attack_type,
                is_cic_2018=is_cic_2018,
                normalization=normalization
            )
            self.initialize_from_one_vs_all_split(splits, attack_label=target_attack_type)
            if inter_dataset_run: # Load inter dataset
                assert inter_dataset_benign_path is not None, "Inter dataset <benign> path must be provided for inter dataset run."
                assert inter_dataset_malicious_path is not None, "Inter dataset <malicious> path must be provided for inter dataset run."
                validation_splits = self.load_inter_dataset_split(
                    benign_path=inter_dataset_benign_path,
                    malicious_path=inter_dataset_malicious_path,
                    target_attack_type=target_attack_type,
                    is_cic_2018=not is_cic_2018
                )
                self.x_val, self.y_val, self.plain_label_val = validation_splits # overwrite the intra dataset validation set
        else:
            self.load_default_dataset()


    def get_batch(self, batch_size=100) -> Tuple[pd.DataFrame, np.ndarray]:
        indexes = list(range(self.index, self.index + batch_size))
        if max(indexes) > self.shape[0] - 1:
            dif = max(indexes) - self.shape[0]
            indexes[len(indexes) - dif - 1:len(indexes)] = list(range(dif + 1))
            self.index = batch_size - dif
        else:
            self.index += batch_size
        batch = self.x_train.iloc[indexes]
        y = self.y_train[indexes]
        plain_labels = self.train_labels.iloc[indexes]
        return batch, y, plain_labels

    def get_full(self) -> Tuple[pd.DataFrame, np.ndarray]:
        return self.x_train, self.y_train.values if isinstance(self.y_train, pd.Series) else self.y_train

    def get_test_set(self) -> Tuple[pd.DataFrame, np.ndarray]:
        x_test, y_test = self.x_test, self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test
        x_test, y_test = shuffle(x_test, y_test, random_state=42)
        return x_test, y_test

    def get_attack_types(self) -> List[str]:
        """
        Returns all attack types (i.e., abstract categories) found in the test set, based on the original label mapping.
        """
        attack_map = cic_attack_map_one_vs_all if self.one_vs_all else cic_attack_map
        mapped = self.test_labels.map(attack_map)

        unmapped = self.test_labels[mapped.isna()]
        if not unmapped.empty:
                logging.warning(f"⚠️ Warning: Found {len(unmapped)} unmapped label entries:\n{unmapped.value_counts()}")
        return sorted(mapped.dropna().unique())


    def load_formatted_df(self):
        """
        Load the full training data (features + labels) into self.df.

        This method ensures that only training data is loaded into self.df,
        leaving test data untouched for later evaluation.
        """
        if hasattr(self, 'df') and self.df is not None:
            self.loaded = True
            self.index = GLOBAL_RNG.integers(0, self.df.shape[0] - 1, dtype=np.int32)
            return

        # Combine train data and corresponding labels
        self.df = self.x_train.copy()
        self.df["Label"] = self.train_labels.reset_index(drop=True)

        self.loaded = True
        self.index = GLOBAL_RNG.integers(0, self.df.shape[0] - 1, dtype=np.int32)

        self.attack_names = sorted(set(cic_attack_map.keys()))

    def load_one_vs_all_split(
        self,
        benign_path: str,
        malicious_path: str,
        target_attack_type: str = ATTACK_TYPE_DDOS,
        benign_total: int = 250000,
        train_benign_size: int = 50000,
        benign_ratio = 0.8032, # CIC-IDS-2017 Ratio of total Data
        malicious_ratio: float = 0.1343, # CIC-IDS-2017 Ratio of total
        normalization: str = "linear",
        is_cic_2018: bool = False,
        random_state: int = 42
    ):
        # Helper
        def process(df, x_min=None, x_max=None):
            labels = df["OriginalLabel"].copy() if "OriginalLabel" in df.columns else df["Label"].copy()
            y = df["Label"]
            X = df.drop(columns=["Label", "OriginalLabel", "Timestamp", "Destination Port", "Abstract"], errors="ignore")
            if normalization == "linear":
                if x_min is None or x_max is None:
                    x_min = X.min()
                    x_max = X.max()
                x_norm = (X - x_min) / (x_max - x_min)
                return x_norm, y, labels, x_min, x_max

            return X, y, labels, None, None
        file_format = benign_path.split(".")[-1]
        # Lade Daten
        benign_df = pd.read_feather(benign_path) if file_format == "feather" else pd.read_csv(benign_path)
        benign_df["OriginalLabel"] = benign_df["Label"]
        benign_df = benign_df[benign_df["Label"] == "Benign"].sample(n=benign_total, random_state=random_state)

        malicious_df = pd.read_feather(malicious_path) if file_format == "feather" else pd.read_csv(malicious_path)
        malicious_df["Label"] = malicious_df["Label"].apply(normalize_label)
        malicious_df["Abstract"] = malicious_df["Label"].map(cic_attack_map)
        malicious_df["OriginalLabel"] = malicious_df["Label"] 
        malicious_df["Label"] = malicious_df["Abstract"]


        # Split ratios
        test_size_ratio = 0.85 if is_cic_2018 else 0.7

        # Train/Val/Test split benign
        benign_train = benign_df.iloc[:train_benign_size]
        benign_remaining = benign_df.iloc[train_benign_size:]
        benign_val, benign_test = train_test_split(benign_remaining, test_size=test_size_ratio, random_state=random_state)

        # Filter gewünschter Angriff
        attack_df = malicious_df[malicious_df["Abstract"] == target_attack_type]
        if attack_df.empty:
            raise ValueError(f"No samples found for attack type '{target_attack_type}'.")

        # Berechne Samplegröße
        #attack_total = int((benign_total * malicious_ratio) / (1 - malicious_ratio))
        attack_train_count = int(train_benign_size * (malicious_ratio / benign_ratio))

        attack_df = attack_df.sample(frac=1.0, random_state=random_state)  # Shuffle einmal komplett
        attack_train = attack_df.iloc[:attack_train_count]
        remaining_attack_df = attack_df.iloc[attack_train_count:]
        attack_val, attack_test = train_test_split(remaining_attack_df, test_size=test_size_ratio, random_state=random_state)


        # combine
        val_df = pd.concat([benign_val, attack_val], ignore_index=True)
        test_df = pd.concat([benign_test, attack_test], ignore_index=True)
        train_df = pd.concat([benign_train, attack_train], ignore_index=True)

        # process
        x_train, y_train, plain_labels_train, x_min, x_max = process(train_df)
        x_val, y_val, plain_labels_val, _, _ = process(val_df, x_min=x_min, x_max=x_max)
        x_test, y_test, plain_labels_test, _, _ = process(test_df, x_min=x_min, x_max=x_max)

        x_train, y_train, plain_labels_train = shuffle(x_train, y_train, plain_labels_train, random_state=42)
        x_val, y_val, plain_labels_val = shuffle(x_val, y_val, plain_labels_val, random_state=42)
        x_test, y_test, plain_labels_test = shuffle(x_test, y_test, plain_labels_test, random_state=42)

        logging.info(f"One-vs-All Split ready (target: {target_attack_type}):")
        logging.info(f"Train: {x_train.shape}, Validation: {x_val.shape}, Test: {x_test.shape}")

        # Return as Dict
        return {
            "train": (x_train, y_train, plain_labels_train),
            "val": (x_val, y_val, plain_labels_val),
            "test": (x_test, y_test, plain_labels_test)
        }
    

    def load_inter_dataset_split(
        self,
        benign_path: str,
        malicious_path: str,
        target_attack_type: str = ATTACK_TYPE_DDOS,
        is_cic_2018: bool = False,
    ):
        """
        Creates a pure test set from a second dataset for evaluation.
        Uses internally load_one_vs_all_split(), but returs only the test part.
        """
        splits = self.load_one_vs_all_split(
            benign_path=benign_path,
            malicious_path=malicious_path,
            benign_total=550000 if is_cic_2018 else 250000,
            benign_ratio=0.8459 if is_cic_2018 else 0.8032,
            target_attack_type=target_attack_type,
            is_cic_2018=is_cic_2018,
        )
        return splits["test"]


    def initialize_from_one_vs_all_split(self, splits, attack_label=ATTACK_TYPE_DDOS):
        x_train, y_train, plain_label_train = splits["train"]
        x_val, y_val, plain_label_val = splits["val"]
        x_test, y_test, plain_label_test = splits["test"]

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.plain_label_train = plain_label_train
        self.plain_label_val = plain_label_val
        self.plain_label_test = plain_label_test

        self.train_labels = pd.Series(["Benign" if y == 'Benign' else attack_label for y in y_train])
        self.test_labels = pd.Series(["Benign" if y == 'Benign' else attack_label for y in y_test])
        assert len(self.x_train) == len(self.plain_label_train), f"Mismatch: {len(self.x_train)} features vs. {len(self.plain_label_train)} labels"
        assert len(x_train) == len(self.plain_label_train), (f"❌ Label mismatch: {len(x_train)} samples vs {len(self.plain_label_train)} labels")
        assert not pd.Series(self.plain_label_train).isna().any(), "NaNs in plain_label_train!"

        # Baue vollständiges Trainings-DF
        self.df = pd.DataFrame(x_train).reset_index(drop=True).assign(Label=pd.Series(self.plain_label_train).reset_index(drop=True))
        # DataFrame fürs Logging o. Replay

        assert not self.df["Label"].isna().any(), "NaN-Labels present Merge failed!"

        # Dimensions-/Index-Infos
        self.obs_size = self.x_train.shape[1]
        self.shape = self.x_train.shape
        self.index = GLOBAL_RNG.integers(0, self.shape[0] - 1, dtype=np.int32)

        # Infofelder für spätere Agentenlogik
        #self.attack_names = sorted([label for label in cic_attack_map])
        available_labels = set(self.df["Label"].unique())

        # Relevant: Alle Benign + Attack-Typ-spezifischen Labels, die im DF vorkommen
        attack_names = ["Benign"] + [
            name for name in cic_attack_map
            if cic_attack_map[name] == attack_label and name in available_labels
        ]

        self.attack_names = sorted(attack_names)
        self.attack_types = ["Benign", attack_label]
        self.attack_map = cic_attack_map
        self.all_attack_names = sorted(set(cic_attack_map.keys()))
        self.loaded = True
        logging.info(f"[Init from split] Loaded {self.x_train.shape[0]} train, {self.x_test.shape[0]} test samples.")


    def load_default_dataset(self): # 
        x_train, x_test, y_train, y_test, _train_labels, _test_labels = load_data(
            data_file_path=self.benign_path,
            train_size=50000,
            test_size=200000,
            downsample_benign=True,
            benign_sample_size=550000 if self.is_cic_2018_training_set else 250000,
        )
        x_fraud, y_fraud, labels_fraud = load_data_fraud(data_file_path=self.malicious_path)

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = pd.concat([x_test, x_fraud], ignore_index=True)
        self.y_test = pd.concat([y_test, y_fraud], ignore_index=True)

        if self.normalization == 'linear':
            self.x_train = (self.x_train - self.x_train.min()) / (self.x_train.max() - self.x_train.min())
            self.x_test = (self.x_test - self.x_train.min()) / (self.x_train.max() - self.x_train.min())

        assert self.x_train.shape[0] == self.y_train.shape[0], "Anzahl Zeilen passt nicht!"

        self.train_labels = _train_labels.reset_index(drop=True)
        self.test_labels = pd.concat([_test_labels, labels_fraud], ignore_index=True)
        self.test_labels = self.test_labels.apply(normalize_label)
        self.test_labels_original = self.test_labels

        self.df = pd.concat([
            self.x_train.reset_index(drop=True),
            self.train_labels.reset_index(drop=True).rename("Label")
        ], axis=1)

        assert not self.df["Label"].isna().any(), "NaN-Labels present – Merge failed!"

        self.obs_size = self.x_train.shape[1]
        self.shape = self.x_train.shape
        self.index = GLOBAL_RNG.integers(0, self.shape[0] - 1, dtype=np.int32)

        if self.one_vs_all:
            self.y_train = self.train_labels.map(cic_attack_map_one_vs_all)
            self.y_test = self.test_labels.map(cic_attack_map_one_vs_all)
            self.attack_map = cic_attack_map_one_vs_all
        else:
            self.y_train = y_train.reset_index(drop=True)
            self.y_test = pd.concat([y_test, y_fraud], ignore_index=True)
            self.attack_map = cic_attack_map

        self.attack_names = sorted(set(self.test_labels.unique()))
        self.attack_types = self.get_attack_types()
        self.all_attack_names = list(self.attack_map.keys())
        self.loaded = True

        logging.info(f"CICDataManager: {self.x_train.shape[0]} training samples, {self.x_test.shape[0]} test samples loaded.")
        logging.info(f"CICDataManager: {self.x_train.shape[1]} features loaded.")