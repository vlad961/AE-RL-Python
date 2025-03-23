import logging
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from typing import Dict, List, Tuple

cwd = os.getcwd()
data_root_dir = os.path.join(cwd, "data/datasets/")
data_original_dir = os.path.join(data_root_dir, "origin-kaggle-com/nsl-kdd/")
data_formated_dir = os.path.join(data_root_dir, "formated/")
formated_train_path = os.path.join(data_formated_dir, "balanced_training_data.csv") #  formated_train_adv
formated_test_path = os.path.join(data_formated_dir, "balanced_test_data.csv") #  formated_test_adv
kdd_train = os.path.join(data_original_dir, "KDDTrain+.txt")
kdd_test = os.path.join(data_original_dir, "KDDTest+.txt")

attack_map: Dict[str, str] = {'normal': 'normal',

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

col_names: List[str] = ["duration", "protocol_type", "service", "flag", "src_bytes",
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

class DataCls:
    def __init__(self, trainset_path: str = kdd_train, testset_path: str = kdd_test, formated_trainset_path: str = formated_train_path, 
                 formated_testset_path: str = formated_test_path, dataset_type: str = "train"):
        """
        Initialize the DataCls object with the given parameters.
        Initialize the attack types, attack names, and attack map.
        Initialize the DataFrame and formats it if not already done, column names, and index.
        Initializes the present attack_names in the DataFrame.
        Args:
            trainset_path (str): Path to the training dataset.
            testset_path (str): Path to the test dataset.
            formated_trainset_path (str): Path to formated training dataset output.
            formated_testset_path (str): Path to formated test dataset output.
            dataset_type (str): Type of the dataset, either 'train' or 'test'. Default is 'train'.
        """
        self.col_names = col_names
        self.index = 0
        # Data formated path and test path.
        self.loaded = False
        self.dataset_type = dataset_type
        self.trainset_path = trainset_path
        self.testset_path = testset_path

        self.formated_train_path = formated_trainset_path
        self.formated_test_path = formated_testset_path

        self.attack_types = attack_types
        self.attack_names = []
        self.attack_map = attack_map
        self.all_attack_names = list(self.attack_map.keys())
        self.format_data_instance()
        self.update_attack_names()

    def get_shape(self):
        if self.loaded is False:
            self.load_formatted_df()

        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape

    ''' Get n-rows from loaded data
        The dataset must be loaded in RAM
    '''

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
        if max(indexes) > self.get_shape()[0] - 1:
            dif = max(indexes) - self.data_shape[0]
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

        Args:
            None

        Returns:
            None

        Side Effects:
            - Sets self.df to the loaded DataFrame.
            - Sets self.index to a random integer within the range of the DataFrame's row count.
            - Sets self.loaded to True.
            - Updates self.attack_names with the names of the attacks present in the DataFrame.
        """
        if self.df is not None and self.attack_names is not None and len(self.attack_names) > 0:
            self.loaded = True
            self.index = np.random.randint(0, self.df.shape[0] - 1, dtype=np.int32)
            return
        else:
            if self.dataset_type == 'train':
                self.df = pd.read_csv(self.formated_train_path, sep=',')  # Read again the csv
            else:
                self.df = pd.read_csv(self.formated_test_path, sep=',')
            self.index = np.random.randint(0, self.df.shape[0] - 1, dtype=np.int32)
            self.loaded = True
            # Create a list with the existent attacks in the df
            for att in self.attack_map:
                if att in self.df.columns:
                    # Add only if there exists at least 1
                    if np.sum(self.df[att].values) >= 1 and att not in self.attack_names:
                        self.attack_names.append(att)

    @staticmethod
    def format_data(dataset_type: str, trainset_path: str, testset_path: str, formated_train_path: str, formated_test_path: str) -> pd.DataFrame:
        """
        Format the data for ready-to-use.
        
        Details:
            Formating the training dataset for ready-2-use data
            Remove the difficulty column because we do not want to consider it, and add the one-hot encoding for the categorical columns
            Further su_attempted values are changed from 2 to 0.
             
            Note: Upon my research, in the original KDD dataset there were only 0 and 1 values for the su_attempted column. 
            The improved NSL-KDD dataset has 0, 1, and 2 values but 2 values were not mentioned in the description.
            Therefore, I assume the authors of AE-RL considered 2 values as mistakes and changed them to 0.
            Reference to original KDD-Data: https://kdd.ics.uci.edu/databases/kddcup99/task.html
            Reference to NSL-KDD originate work was done by Tavallaee et. al. from University of New Brunswick how ever the data doesnt seem to be further maintained.
              https://www.unb.ca/cic/datasets/nsl.html(dead link),
            Alternative links to the NSL-KDD dataset:
             https://www.kaggle.com/datasets/hassan06/nslkdd (This project is based on this dataset)
             https://ieee-dataport.org/documents/nsl-kdd-0#files
        Args:
            trainset_path (str): Path to the training dataset.
            testset_path (str): Path to the test dataset.
            formated_train_path (str): Path to the formatted training dataset.
            formated_test_path (str): Path to the formatted test dataset.
            col_names (list): List of column names for the dataset.

        Returns:
            None
        """
        if os.path.exists(formated_train_path) and os.path.exists(formated_test_path):
            if(dataset_type == 'train'):
                logging.info(f"Reading formated train data from: {formated_train_path}")
                return pd.read_csv(formated_train_path, sep=',')
            else:
                logging.info(f"Reading formated test data from: {formated_test_path}")
                return pd.read_csv(formated_test_path, sep=',')
        
        # Format the data
        formated_dir = os.path.dirname(formated_train_path)
        if not os.path.exists(formated_dir):
            os.makedirs(formated_dir)

        # Formating the training dataset
        df = pd.read_csv(trainset_path, sep = ',', names = col_names, index_col = False)
        if 'difficulty' in df.columns:
            df.drop('difficulty', axis=1, inplace=True)

        test_data = pd.read_csv(testset_path, sep = ',', names = col_names, index_col = False)
        if 'difficulty' in test_data:
            test_data.drop('difficulty', axis = 1, inplace = True)

        amount_train_samples = df.shape[0] # Save the amount of training samples
        frames = [df, test_data]
        df = pd.concat(frames) # Concatenate the train and test dataframes

        # Dataframe processing
        # One hot encoding for categorical columns
        df = pd.concat([df.drop('protocol_type', axis=1), pd.get_dummies(df['protocol_type'])], axis=1)
        df = pd.concat([df.drop('service', axis=1), pd.get_dummies(df['service'])], axis=1)
        df = pd.concat([df.drop('flag', axis=1), pd.get_dummies(df['flag'])], axis=1)

        # NSL-KDD seems to have introduced faulty su_attempted values of '2' which are not documented in the original NSL nor in the improved NSL-KDD work.
        # Therefore, I assume the authors of AE-RL considered 2 as mistakes and changed them to 0)
        # 1 if ``su root'' command attempted; 0 otherwise
        df['su_attempted'] = df['su_attempted'].replace(2.0, 0.0)

        # One hot encoding for labels
        df = pd.concat([df.drop('labels', axis=1), pd.get_dummies(df['labels'])], axis=1)

        # Normalization of the df
        # Normalization of the continous columns in the df (0-1) # TODO: Check if I get an improvement if I normalize the data with the tensorflow functional API instead. # TODO: What if I use float32 instead of float64?
        for indx, dtype in df.dtypes.items():
            if dtype == 'float64' or dtype == 'int64':
                if df[indx].max() == 0 and df[indx].min() == 0:
                    df[indx] = 0.0 # dtype = float64 TODO: change to float32 since there is anyway no information in the column
                else:
                    df[indx] = (df[indx] - df[indx].min()) / (df[indx].max() - df[indx].min()) # dtype = float64

        # Save data
        test_df = df.iloc[amount_train_samples:df.shape[0]]
        test_df = shuffle(test_df, random_state=np.random.randint(0, 100))
        df = df[:amount_train_samples]
        df = shuffle(df, random_state=np.random.randint(0, 100))
        test_df.to_csv(formated_test_path, sep=',', index=False)
        df.to_csv(formated_train_path, sep=',', index=False)
        logging.info(f"Formated train data saved in: {formated_train_path}")
        logging.info(f"Formated test data saved in: {formated_test_path}")
        return df

    def format_data_instance(self) -> None:
        """
        Instance method to format the data using instance attributes.
        """
        self.df = DataCls.format_data(self.dataset_type, self.trainset_path, self.testset_path, self.formated_train_path, self.formated_test_path)
        self.loaded = True


    @staticmethod
    def calculate_obs_size(data_path) -> int:
        """
        Calculate the observation size based on the given data path.

        This method calculates the observation size, which is the number of columns in the DataFrame
        minus the number of all attack names.

        Args:
            data_path (str): Path to the already formatted data file. 

        Returns:
            int: The observation size.

        Raises:
            FileNotFoundError: If the formatted data file does not exist at the given path.
        """
        if (not os.path.exists(data_path)):
            raise FileNotFoundError(f"File {data_path} not found. Please check the path.")
            return -1
        
        df = pd.read_csv(data_path, sep=',')
        obs_size = df.shape[1] - len(list(attack_map.keys()))
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
        for att in attack_map:
            if att in df.columns:
                # Add only if there exists at least 1
                if np.sum(df[att].values) >= 1 and att not in attack_names:
                    attack_names.append(att)
        return attack_names

    def update_attack_names(self):
        """
        Update the list of existing attacks in the DataFrame.
        """
        self.attack_names = []
        for att in self.attack_map:
            if att in self.df.columns:
                # Add only if there exists at least 1 attack in the column
                if np.sum(self.df[att].values) >= 1 and att not in self.attack_names:
                    self.attack_names.append(att)

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
        for att in attack_names:
            if att in self.df.columns:
                # Add only if there exists at least 1 attack in the column
                if np.sum(self.df[att].values) >= 1 and att not in self.attack_names:
                    self.attack_names.append(att)

    def get_balanced_samples(self) -> pd.DataFrame:
        """
        Get equally balanced data for all attack types.

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
        self.update_attack_names_for_given_attacks(all_attack_names)
        return self.df, self.attack_names