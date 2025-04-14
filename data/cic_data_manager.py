import logging
import pandas as pd
import numpy as np
from data.datasets.cic_ids.data_loader import load_data, load_data_fraud
from typing import List, Tuple
from utils.config import GLOBAL_RNG

cic_attack_map = {
    'Benign': 'Benign',
    'Bot': 'Botnet',
    'DDOS attack-LOIC-UDP': '(D)DOS',
    'DDoS': '(D)DOS',
    'DDoS attacks-LOIC-HTTP': '(D)DOS',
    'DDOS attack-HOIC': '(D)DOS',
    'DoS attacks-Hulk': '(D)DOS',
    'DoS attacks-GoldenEye': '(D)DOS',
    'DoS attacks-Slowloris': '(D)DOS',
    'DoS attacks-SlowHTTPTest': '(D)DOS',
    'DoS Hulk': '(D)DOS',
    'DoS GoldenEye': '(D)DOS',
    'DoS slowloris': '(D)DOS',
    'DoS Slowhttptest': '(D)DOS',
    'PortScan': 'Probe',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'FTP-BruteForce': 'Brute Force',
    'SSH-Bruteforce': 'Brute Force',
    'Web Attack  Brute Force': 'Web Attack',
    'Web Attack  XSS': 'Web Attack',
    'Web Attack  Sql Injection': 'Web Attack',
    'Brute Force -Web': 'Web Attack',
    'Brute Force -XSS': 'Web Attack',
    'SQL Injection': 'Web Attack',
    'Infilteration': 'Infiltration', #CICIDS2018
    'Infiltration': 'Infiltration', #CICIDS2017
    'Heartbleed': 'R2L',
}

import re

@staticmethod
def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return label
    # Ersetze alle Sonderzeichen und vereinheitliche Whitespace
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
    def __init__(self, benign_path: str, malicious_path: str,  cic_2017: bool, normalization: str = 'linear',  one_vs_all: bool = False):
        self.one_vs_all = one_vs_all
        x_train, x_test, y_train, y_test, _train_labels, _test_labels = load_data(
            data_file_path=benign_path,
            train_size=50000,
            test_size=200000,
            downsample_benign=True,
            benign_sample_size=250000 if cic_2017 else 550000,
        )
        x_fraud, y_fraud, labels_fraud = load_data_fraud(data_file_path=malicious_path)

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = pd.concat([x_test, x_fraud], ignore_index=True)
        self.y_test = pd.concat([y_test, y_fraud], ignore_index=True)

        if normalization == 'linear':
            self.x_train = (self.x_train - self.x_train.min()) / (self.x_train.max() - self.x_train.min())
            self.x_test = (self.x_test - self.x_train.min()) / (self.x_train.max() - self.x_train.min())

        assert self.x_train.shape[0] == self.y_train.shape[0], "Anzahl Zeilen passt nicht!"
        
        self.train_labels = _train_labels.reset_index(drop=True)
        self.test_labels = pd.concat([_test_labels, labels_fraud], ignore_index=True)
        self.test_labels = self.test_labels.apply(normalize_label)
        # Originale Labels für spätere Auswertung sichern
        self.test_labels_original = self.test_labels
        #self.df = pd.concat([self.x_train, self.y_train], axis=1)
        # Sicherer Merge:
        self.df = pd.concat([
            self.x_train.reset_index(drop=True),
            #self.y_train.reset_index(drop=True).rename("Label")
            self.train_labels.reset_index(drop=True).rename("Label")
        ], axis=1)

        # Prüfung:
        assert not self.df["Label"].isna().any(), "NaN-Labels present – Merge failed!"

        self.obs_size = self.x_train.shape[1]
        self.shape = self.x_train.shape
        self.index = GLOBAL_RNG.integers(0, self.shape[0] - 1, dtype=np.int32)

        if one_vs_all:
            self.y_train = self.train_labels.map(cic_attack_map_one_vs_all)
            self.y_test = self.test_labels.map(cic_attack_map_one_vs_all)
            self.attack_names = sorted(set(self.test_labels.unique()))
            self.attack_map = cic_attack_map_one_vs_all
        else:
            self.y_train = y_train.reset_index(drop=True)
            self.y_test = pd.concat([y_test, y_fraud], ignore_index=True)
            self.attack_names = sorted(set(self.test_labels.unique()))
            self.attack_map = cic_attack_map

        self.attack_types: List[str] = self.get_attack_types()
        self.loaded = True
        self.all_attack_names = list(self.attack_map.keys()) # Vermutlich zu viele Attacken. Einmal ein ordentliches Mapping aus den Daten selbst heraus lesen.
        
        logging.info(f"CICDataManager: {self.x_train.shape[0]} training samples, {self.x_test.shape[0]} test samples loaded.")
        logging.info(f"CICDataManager: {self.x_train.shape[1]} features loaded.")

    def get_batch(self, batch_size=100) -> Tuple[pd.DataFrame, np.ndarray]: # FIXME: hier werden mehr als batch_size zurückgegeben siehe ursprüngliche Implementierung und transferiere hier her.
        indexes = list(range(self.index, self.index + batch_size))
        if max(indexes) > self.shape[0] - 1:
            dif = max(indexes) - self.shape[0]
            indexes[len(indexes) - dif - 1:len(indexes)] = list(range(dif + 1))
            self.index = batch_size - dif
        else:
            self.index += batch_size
        batch = self.x_train.iloc[indexes]
        labels = self.y_train.iloc[indexes]
        return batch, labels.values

    def get_full(self) -> Tuple[pd.DataFrame, np.ndarray]:
        return self.x_train, self.y_train.values

    def get_test_set(self) -> Tuple[pd.DataFrame, np.ndarray]:
        return self.x_test, self.y_test.values


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

        # Kombiniere Trainingsdaten (Features und Labels)
        self.df = self.x_train.copy()
        self.df["Label"] = self.train_labels.reset_index(drop=True)

        self.loaded = True
        self.index = GLOBAL_RNG.integers(0, self.df.shape[0] - 1, dtype=np.int32)

        # Attack names = konkrete Klassenbezeichner im Label
        self.attack_names = sorted(self.df["Label"].unique())


""" FIXME: Diese Methode lädt test daten aber aktuell soll nur auf normal trainiert werden.
    def load_formatted_df(self):
        
        Load the full benign + malicious test data (already processed & downsampled) into self.df.

        This method is equivalent zur NSL-KDD-Version – sie lädt die aggregierten Daten (z. B. .csv/.feather)
        aus dem vorher bereinigten Speicherpfad und setzt alle nötigen Felder wie `.df`, `.index`, `.loaded`, `.attack_names`.

        Voraussetzung:
            - self.test_labels muss korrekt gesetzt sein
            - self.x_test muss existieren
        
        if hasattr(self, 'df') and self.df is not None:
            self.loaded = True
            self.index = GLOBAL_RNG.integers(0, self.df.shape[0] - 1, dtype=np.int32)
            return

        # Kombiniere Features und Labels für den Zugriff
        self.df = self.x_test.copy()
        self.df["Label"] = self.test_labels.reset_index(drop=True)

        self.loaded = True
        self.index = GLOBAL_RNG.integers(0, self.df.shape[0] - 1, dtype=np.int32)

        # Attack names = konkrete Klassenbezeichner im Label
        self.attack_names = sorted(self.df["Label"].unique())"""