import logging
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from utils.config import GLOBAL_RNG

nsl_kdd_attack_map: Dict[str, str] = {
                'normal': 'normal',
                'back': 'DoS',
                'land': 'DoS',
                'neptune': 'DoS',
                'pod': 'DoS',
                'smurf': 'DoS',
                'teardrop': 'DoS',
                'mailbomb': 'DoS',
                'apache2': 'DoS',
                'processtable': 'DoS',
                'udpstorm': 'DoS',

                'ipsweep': 'Probe',
                'nmap': 'Probe',
                'portsweep': 'Probe',
                'satan': 'Probe',
                'mscan': 'Probe',
                'saint': 'Probe',

                'ftp_write': 'R2L',
                'guess_passwd': 'R2L',
                'imap': 'R2L',
                'multihop': 'R2L',
                'phf': 'R2L',
                'spy': 'R2L',
                'warezclient': 'R2L',
                'warezmaster': 'R2L',
                'sendmail': 'R2L',
                'named': 'R2L',
                'snmpgetattack': 'R2L',
                'snmpguess': 'R2L',
                'xlock': 'R2L',
                'xsnoop': 'R2L',
                'worm': 'R2L',

                'buffer_overflow': 'U2R',
                'loadmodule': 'U2R',
                'perl': 'U2R',
                'rootkit': 'U2R',
                'httptunnel': 'U2R',
                'ps': 'U2R',
                'sqlattack': 'U2R',
                'xterm': 'U2R'
}

nsl_kdd_col_names: List[str] = ["duration", "protocol_type", "service", "flag", "src_bytes",
                     "dst_bytes", "land_f", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                     "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                     "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                     "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                     "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                     "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                     "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels", "difficulty"]

attack_types: List[str] = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']

class NslKddDataManager:
    def __init__(self, trainset_path: str, testset_path: str, formated_trainset_path: str,
                 formated_testset_path: str, dataset_type: str = "train", dataset_name: str = "nsl-kdd", normalization: str = 'linear', **kwargs):
        self.col_names = nsl_kdd_col_names
        self.index: int = 0
        # Data formated path and test path.
        self.loaded = False
        self.dataset_type = dataset_type
        self.trainset_path = trainset_path
        self.testset_path = testset_path
        self.multiple_attackers = kwargs.get('multiple_attackers', False)

        self.formated_train_path = formated_trainset_path
        self.formated_test_path = formated_testset_path

        self.attack_types = attack_types
        
        self.attack_map = nsl_kdd_attack_map
        self.all_attack_names = list(self.attack_map.keys())
        self.df: pd.DataFrame = None
        if self.dataset_type == 'test':
            _, self.df = self.format_nsl_kdd_data_instance(normalization)
        else:
            self.df, _ = self.format_nsl_kdd_data_instance(normalization)
        self.attack_names = NslKddDataManager.update_attack_names(self.attack_map, self.df)
        # Initialize a reusable random generator
        self.loaded = True
        self.shape = self.df.shape
        self.obs_size = self.shape[1] - len(list(nsl_kdd_attack_map.keys())) # Number of columns/features - number of all possible attack names

    def get_batch(self, batch_size=100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get a batch of data from the loaded DataFrame.

        This method reads the rows of the DataFrame starting from the current index up to the current index plus the batch size.
        If the current index plus the batch size exceeds the number of rows in the DataFrame, the method wraps around to the beginning of the DataFrame.
        The method returns the batch of data and the corresponding labels.

        Args:
            batch_size (int): The number of rows to read from the DataFrame.

        Returns:
            tuple: A tuple containing the batch of data and the corresponding labels.
        """
        if self.loaded is False:
            self.load_formatted_df()

        # Read the df rows
        indexes = list(range(self.index, self.index + batch_size))
        if max(indexes) > self.shape[0] - 1:
            dif = max(indexes) - self.shape[0]
            indexes[len(indexes) - dif - 1:len(indexes)] = list(range(dif + 1))
            self.index = batch_size - dif
            batch = self.df.iloc[indexes]
        else:
            batch = self.df.iloc[indexes]
            self.index += batch_size

        labels = batch[self.attack_names]

        batch = batch.drop(self.all_attack_names, axis=1)

        # Laufzeitprüfung
        if not isinstance(batch, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
            raise TypeError("Expected both batch and labels to be pandas DataFrames.")

        return batch, labels

    def get_full(self):
        """
        This method loads the full DataFrame. All possible attack names from the DataFrame are dropped.
        The resulting panda DataDrame is converted into a tensor.
        All labels/attacks that have at least one sample within the original DataFrame are saved, and the result is returned.

        Returns:
            tuple: A tuple containing a tensor of the DataFrame without any features about all the possible attacks and the corresponding labels
        """
        if self.loaded is False:
            self.load_formatted_df()

        labels = self.df[self.attack_names]

        batch = self.df.drop(self.all_attack_names, axis=1)
        return batch, labels

    def load_formatted_df(self):
        """
        Load the DataFrame from the formatted training or test dataset path.

        This method reads the CSV file from the specified path based on the dataset type
        ('train' or 'test'), initializes the DataFrame, and updates the list of existing
        attacks in the DataFrame.

        Side Effects:
            - Sets self.df to the loaded DataFrame.
            - Sets self.index to a random integer within the range of the DataFrame's row count.
            - Sets self.loaded to True.
            - Updates self.attack_names with the names of the attacks present in the DataFrame.
        """
        if self.df is not None and self.attack_names is not None and len(self.attack_names) > 0:
            self.loaded = True
            self.index = GLOBAL_RNG.integers(0, self.shape[0] - 1, dtype=np.int32)
        else:
            if self.dataset_type == 'train':
                self.df = pd.read_csv(self.formated_train_path, sep=',')  # Read again the csv
            else:
                self.df = pd.read_csv(self.formated_test_path, sep=',')
            self.index = GLOBAL_RNG.integers(0, self.shape[0] - 1, dtype=np.int32)
            self.loaded = True
            self.attack_names = NslKddDataManager.update_attack_names(self.attack_map, self.df)

    @staticmethod
    def load_formatted_nsl_kdd_if_exists(formated_train_path: str, formated_test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the formatted NSL-KDD dataset if it exists already.

        Args:
            formated_train_path (str): Path to the formatted training dataset.
            formated_test_path (str): Path to the formatted test dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Formatted training and test datasets.
        """
        if os.path.exists(formated_train_path) and os.path.exists(formated_test_path):
            logging.info(f"Reading formated train data from: {formated_train_path}")
            train_data = pd.read_csv(formated_train_path, sep=',', dtype=np.float32)
            logging.info(f"Reading formated test data from: {formated_test_path}")
            test_data =  pd.read_csv(formated_test_path, sep=',', dtype=np.float32)
            for col in train_data.select_dtypes(include=[np.number]).columns:
                train_data[col] = train_data[col].astype(np.float32)

            for col in test_data.select_dtypes(include=[np.number]).columns:
                test_data[col] = test_data[col].astype(np.float32)

            return train_data, test_data
        else:
            logging.info(f"Formated 'train' data not found in: {formated_train_path}")
            logging.info(f"Formated 'test' data not found in: {formated_test_path}")
            return None, None


    @staticmethod
    def get_formated_nsl_kdd_data(trainset_path: str, testset_path: str, formated_train_path: str, formated_test_path: str, normalization: str = 'linear') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return the formatted NSL-KDD dataset if it exists already, otherwise format the original dataset and return the ready for training dataset.
        Format the NSL-KDD dataset to be ready for training.
        
        Details:
            - Remove the difficulty column because we do not want to consider it.
            - Add one-hot encoding for all categorical columns.
            - Further su_attempted values are changed from 2 to 0.
             
            Note: Upon my research, in the original KDD dataset there were only 0 and 1 values for the su_attempted column. 
            The improved NSL-KDD dataset has 0, 1, and 2 values but 2 values were not mentioned in the description.
            Therefore, I assume the authors of AE-RL considered 2 values as mistakes and changed them to 0.
            Reference to original KDD-Data: https://kdd.ics.uci.edu/databases/kddcup99/task.html
            Reference to NSL-KDD originate work was done by Tavallaee et. al. from University of New Brunswick. The data doesnt seem to be further maintained.
              https://www.unb.ca/cic/datasets/nsl.html(dead link),
            Alternative links to the NSL-KDD dataset:
             https://www.kaggle.com/datasets/hassan06/nslkdd (This project is based on this dataset)
             https://ieee-dataport.org/documents/nsl-kdd-0#files
        Args:
            trainset_path (str): Path to the training dataset.
            testset_path (str): Path to the test dataset.
            formated_train_path (str): Path to the formatted training dataset.
            formated_test_path (str): Path to the formatted test dataset.

        Returns:
            Tuple [pd.DataFrame, pd.DataFrame]: Formatted training and test datasets.
        """
        # Load formated data if it exists already
        train, test = NslKddDataManager.load_formatted_nsl_kdd_if_exists(formated_train_path, formated_test_path)
        if train is not None and test is not None:
            return train, test

        # Create Folder if it does not exist
        NslKddDataManager.create_formated_data_dir_if_not_exists(formated_train_path)

        # Formating the training dataset
        train_data = pd.read_csv(trainset_path, sep = ',', names = nsl_kdd_col_names, index_col = False)
        test_data = pd.read_csv(testset_path, sep = ',', names = nsl_kdd_col_names, index_col = False)
        if 'difficulty' in train_data.columns:
            train_data = train_data.drop('difficulty', axis=1)
        if 'difficulty' in test_data:
            test_data = test_data.drop('difficulty', axis = 1)

        amount_train_samples = train_data.shape[0] # Save the amount of training samples
        frames = [train_data, test_data]
        full_data = pd.concat(frames) # Concatenate the train and test dataframes

        # One hot encoding for categorical columns
        full_data = pd.concat([full_data.drop('protocol_type', axis=1), pd.get_dummies(full_data['protocol_type'])], axis=1)
        full_data = pd.concat([full_data.drop('service', axis=1), pd.get_dummies(full_data['service'])], axis=1)
        full_data = pd.concat([full_data.drop('flag', axis=1), pd.get_dummies(full_data['flag'])], axis=1)

        # NSL-KDD seems to have introduced faulty su_attempted values of '2' which are not documented in the original NSL nor in the improved NSL-KDD work.
        # Therefore, I assume the authors of AE-RL considered 2 as mistakes and changed them to 0). 1 if ``su root'' command attempted; 0 otherwise
        full_data['su_attempted'] = full_data['su_attempted'].replace(2.0, 0.0)

        # One hot encoding for labels
        full_data = pd.concat([full_data.drop('labels', axis=1), pd.get_dummies(full_data['labels'])], axis=1)

        # Normalization of the df
        bool_cols = full_data.select_dtypes(include='bool').columns
        full_data[bool_cols] = full_data[bool_cols].astype('float32')
        # Normalization of the continous columns in the df (0-1) //TODO: Check if I get an improvement if I use log normalization instead of linear normalization!!!
        full_data = NslKddDataManager.normalize_numerical_columns(full_data, normalization=normalization)
        logging.info(f"DataManager<Datatypes>:{full_data.dtypes.value_counts()}")
        assert all(np.issubdtype(dt, np.number) for dt in full_data.dtypes), "Non-numerical columns contained. Please make sure the data is normalized correctly!"
        assert all(full_data.dtypes == np.float32), "Not all columns are float32!"

        # Save data
        formatted_test_df = full_data.iloc[amount_train_samples:full_data.shape[0]]
        formatted_train_df = full_data[:amount_train_samples]
        formatted_test_df.to_csv(formated_test_path, sep=',', index=False)
        formatted_train_df.to_csv(formated_train_path, sep=',', index=False)
        logging.info(f"Formated train data saved in: {formated_train_path}")
        logging.info(f"Formated test data saved in: {formated_test_path}")
        return formatted_train_df, formatted_test_df

    def format_nsl_kdd_data_instance(self, normalization: str = 'linear') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Format the NSL-KDD dataset to be ready for training.

        This method loads the NSL-KDD dataset, formats it by removing unnecessary columns, and applies one-hot encoding to categorical columns.
        It also normalizes (linear) the continuous columns and saves the formatted dataset to the specified paths.

        Returns:
            The method returns the formatted training and test datasets.
        """
        train_data, test_data = NslKddDataManager.get_formated_nsl_kdd_data(self.trainset_path, self.testset_path, self.formated_train_path, self.formated_test_path, normalization=normalization)
        return train_data, test_data

    @staticmethod
    def get_obs_size_nsl_kdd(data_path) -> int:
        """
        Calculate the observation size of the nsl-kdd dataset on the given data path.

        This method calculates the observation size, which is the number of columns in the DataFrame
        minus the number of all occuring attack names.

        Args:
            data_path (str): Path to the already formatted data file. 

        Returns:
            int: The observation size.

        Raises:
            FileNotFoundError: If the formatted data file does not exist at the given path.
        """
        if (not os.path.exists(data_path)):
            raise FileNotFoundError(f"File {data_path} not found. Please check the path.")
        
        df = pd.read_csv(data_path, sep=',')
        obs_size = df.shape[1] - len(list(nsl_kdd_attack_map.keys()))
        return obs_size
    
    @staticmethod
    def get_attack_names(data_path) -> list:
        """
        Get the list of names of existing attacks in the DataFrame that is loaded from the given data path.
        An attack is considered to exist if there is at least one instance of it in the DataFrame.

        Args:
            data_path (str): Path to the already formatted data file.

        Returns:
            list: List of names of existing attacks in the DataFrame.
        """
        attack_names = []
        if (not os.path.exists(data_path)):
            raise FileNotFoundError(f"File {data_path} not found. Please check the path.")
        
        df = pd.read_csv(data_path, sep=',')
        # Create a list with the existent attacks in the data frame
        for att in nsl_kdd_attack_map:
            if att in df.columns:
                # Add only if there exists at least 1
                if np.sum(df[att].values) >= 1 and att not in attack_names:
                    attack_names.append(att)
        return attack_names

    @staticmethod
    def update_attack_names(attack_map, df: pd.DataFrame) -> list[str]:
        """
        Update the list of existing attacks in the DataFrame.
        """
        attack_names = []
        for att in attack_map:
            if att in df.columns:
                # Add only if there exists at least 1 attack in the column
                if np.sum(df[att].values) >= 1 and att not in attack_names:
                    attack_names.append(att)
                
        return attack_names

    def get_samples_for_attack(self, attack_name: str, num_samples: int) -> pd.DataFrame:
        """
        Get a specified number of samples for a given attack name.

        This method filters the DataFrame for the given attack name and returns the specified number of samples.
        If the number of samples is 0, the method returns all samples for the given attack type.

        Args:
            attack_name (str): The name of the attack.
            num_samples (int): The number of samples to return.

        Returns:
            pd.DataFrame: The DataFrame containing the specified number of samples for the given attack name.
        """
        if self.loaded is False:
            self.load_formatted_df()

        attack_df = self.df[self.df[attack_name] == 1]
        if num_samples == 0:
            return attack_df
        else:
            return attack_df.head(num_samples)
        

    def get_samples_for_attack_type(self, att_types: List[str], num_samples: int) -> Tuple[pd.DataFrame, list]:
        """
        Get a specified number of samples for a given attack types.

        This method filters the DataFrame for the given attack type and returns the specified number of samples and the attack names.
        If the number of samples is 0, the method returns all samples for the given attack types.

        Side Effects:
            - Updates self.attack_names with the names of the attacks present in the DataFrame.
            - Updates self.df with the filtered DataFrame.
        Args:
            attack_type (List[str]): The types of the attacks.
            num_samples (int): The number of samples to return.

        Returns:
            pd.DataFrame: The DataFrame containing the specified number of samples for the given attack types.
        """
        if self.loaded is False:
            self.load_formatted_df()

        attack_names = []
        for att_typ in att_types:
            attack_names.extend([key for key, value in self.attack_map.items() if value == att_typ])
            
        attack_df = self.df[self.df[attack_names].any(axis=1)]
        self.df = attack_df
        self.update_attack_names_for_given_attacks(attack_names)
        if num_samples == 0:
            return attack_df, self.attack_names
        else:
            return attack_df.head(num_samples), self.attack_names
        
    def update_attack_names_for_given_attacks(self, attack_names):
        """
        Update the list self.attack_names for existing attacks in the DataFrame for the given attack names.

        Args:
            attack_names (list): List of attack names to consider in the DataFrame.
        """
        self.attack_names = []
        valid_columns = []
        self.attacks_not_in_training = []
        for att in attack_names:
            if att in self.df.columns:
                # Check if the column contains at least one sample
                if self.df[att].sum() > 0:
                    valid_columns.append(att)
                else:
                    logging.warning(f"Attack name {att} not found in DataFrame columns.")
                    self.attacks_not_in_training.append(att)
                # Add only if there exists at least 1 attack in the column
                if np.sum(self.df[att].values) >= 1 and att not in self.attack_names:
                    self.attack_names.append(att)


    
###########
# Helpers #
###########

    @staticmethod
    def create_formated_data_dir_if_not_exists(path: str) -> None:
        """
        Create the formatted data directory if it does not exist.

        Args:
            path (str): Path to the directory to be created.

        Returns:
            None
        """
        formated_data_dir = os.path.dirname(path)
        if not os.path.exists(formated_data_dir):
            os.makedirs(formated_data_dir)
            logging.info(f"Created directory: {formated_data_dir}")
        else:
            logging.info(f"Directory already exists: {formated_data_dir}")

    def get_balanced_samples(self) -> pd.DataFrame:
        """
        Downsample the normal class to balance the dataset.
        This method takes 58630 instances of the normal class and 58630 instances of the rest of the attack types.

        Note: there are 125973 instances in the NSL-KDD dataset.
        67343 instances are normal class.
        The number of instances for the rest of the attack types is as follows:
        - DoS: 45927
        - Probe: 11656
        - R2L: 995
        - U2R: 52
        Which sums up to 58630 instances.
        Therefore this method takes 58630 instances of the normal class and 58630 instances of the rest of the attack types.

        Args:
            num_samples (int): The number of samples for each attack type.

        Returns:
            pd.DataFrame: The DataFrame containing the specified number of samples for each attack type.
        """
        if self.loaded is False:
            self.load_formatted_df()

        # Load all data of "normal" class
        normal_df = self.df[self.df['normal'] == 1]

        # Shuffle and take only the first 58630 of 67343 instances of the normal class to balance the data. 
        normal_df = normal_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
        normal_df = normal_df.head(58630)

        # Initialize an empty DataFrame to store the balanced samples
        balanced_df = normal_df

        # Get samples for the rest of the attack types
        all_attack_names = ['normal']
        for att_typ in attack_types:
            if att_typ == 'normal':
                continue
            attack_names = [key for key, value in self.attack_map.items() if value == att_typ]
            attack_df = self.df[self.df[attack_names].any(axis=1)]
            balanced_df = pd.concat([balanced_df, attack_df])
            all_attack_names.extend(attack_names)

        self.df = balanced_df
        self.df = self.df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame one more time
        self.shape = self.df.shape
        self.update_attack_names_for_given_attacks(all_attack_names)
        return self.df, self.attack_names
    
    @staticmethod
    def normalize_numerical_columns(full_data: pd.DataFrame, normalization: str = 'linear') -> pd.DataFrame:
        for col in full_data.select_dtypes(include=[np.number]).columns:
            col_max, col_min = full_data[col].max(), full_data[col].min()

            if col_max == 0 and col_min == 0:
                full_data[col] = 0.0  # Column contains only 0 values.
            elif col_max != col_min:  # Avoid division by zero
                if normalization == 'linear':
                    full_data[col] = (full_data[col] - col_min) / (col_max - col_min)
                elif normalization == 'log':
                    full_data[col] = np.log(full_data[col].clip(lower=1e-10))  # Avoid log(0)

            full_data[col] = full_data[col].astype(np.float32) # Make sure all columns are float32!

        return full_data