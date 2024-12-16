import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle

attack_map = {'normal': 'normal',
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

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                     "dst_bytes", "land_f", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                     "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                     "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                     "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                     "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                     "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                     "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels", "dificulty"]

class DataCls:
    def __init__(self, trainset_path, testset_path, formated_trainset_path, 
                 formated_testset_path, dataset_type="train"):
        """
        Initialize the DataCls object.

        Args:
            trainset_path (str): Path to the training dataset.
            testset_path (str): Path to the test dataset.
            formated_trainset_path (str): Path to the formated training dataset.
            formated_testset_path (str): Path to the formated test dataset.
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

        self.attack_types = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
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

    def get_batch(self, batch_size=100):
        if self.loaded is False:
            self.load_formatted_df()

        # Read the df rows
        indexes = list(range(self.index, self.index + batch_size))
        if max(indexes) > self.data_shape[0] - 1:
            dif = max(indexes) - self.data_shape[0]
            indexes[len(indexes) - dif - 1:len(indexes)] = list(range(dif + 1))
            self.index = batch_size - dif
            batch = self.df.iloc[indexes]
        else:
            batch = self.df.iloc[indexes]
            self.index += batch_size

        labels = batch[self.attack_names]

        batch = batch.drop(self.all_attack_names, axis=1)

        return batch, labels

    def get_full(self):
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
                if np.sum(self.df[att].values) > 1 and att not in self.attack_names:
                    self.attack_names.append(att)
        # self.headers = list(self.df)

    @staticmethod
    def format_data(trainset_path: str, testset_path: str, formated_train_path: str, formated_test_path: str) -> pd.DataFrame:
        """
        Format the data for ready-to-use.
        
        Details:
            Formating the training dataset for ready-2-use data
            Remove the difficulty column because we do not want to consider it, and add the one-hot encoding for the categorical columns
            Further su_attempted is changed to 0 if it is 2.
             
            Note: Upon research my research, in the original KDD dataset there were only 0 and 1 values. 
            The improved NSL-KDD dataset has 0, 1, and 2 values but 2 values were not mentioned in the description.
            Therefore, I assume the authors of AE-RL considered 2 as mistakes and changed it to 0.
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
        # Test if formatted data exists already.
        if os.path.exists(formated_train_path) and os.path.exists(formated_test_path):
            return pd.read_csv(formated_train_path, sep=',')
        # Format the data
        # Get the parent directory of formated_train_path
        formated_dir = os.path.dirname(formated_train_path)
        if not os.path.exists(formated_dir):
            os.makedirs(formated_dir)

        # Formating the training dataset
        df = pd.read_csv(trainset_path, sep = ',', names = col_names, index_col = False)
        if 'dificulty' in df.columns:
            df.drop('dificulty', axis=1, inplace=True)  # in case of difficulty

        test_data = pd.read_csv(testset_path, sep = ',', names = col_names, index_col = False)
        if 'dificulty' in test_data:
            test_data.drop('dificulty', axis = 1, inplace = True)

        amount_train_samples = df.shape[0] # Save the amount of training samples
        frames = [df, test_data]
        df = pd.concat(frames) # Concatenate the train and test dataframes

        # Dataframe processing
        # One hot encoding for categorical columns
        df = pd.concat([df.drop('protocol_type', axis=1), pd.get_dummies(df['protocol_type'])], axis=1)
        df = pd.concat([df.drop('service', axis=1), pd.get_dummies(df['service'])], axis=1)
        df = pd.concat([df.drop('flag', axis=1), pd.get_dummies(df['flag'])], axis=1)

        # NSL-KDD seems to have introduced faulty su_attempted values of '2' which are not documented in the original NSL nor in the improved NSL-KDD work # TODO Lese das Paper nochmal komplett durch und überprüfe ob Aussage tatsächlich stimmt.
        # Therefore, I assume the authors of AE-RL considered 2 as mistakes and changed it to 0)
        # 1 if ``su root'' command attempted; 0 otherwise
        df['su_attempted'] = df['su_attempted'].replace(2.0, 0.0)

        # One hot encoding for labels
        df = pd.concat([df.drop('labels', axis=1), pd.get_dummies(df['labels'])], axis=1)

        # Normalization of the df
        # normalized_df=(df-df.mean())/df.std()
        for indx, dtype in df.dtypes.items():
            if dtype == 'float64' or dtype == 'int64':
                if df[indx].max() == 0 and df[indx].min() == 0:
                    df[indx] = 0.0
                else:
                    df[indx] = (df[indx] - df[indx].min()) / (df[indx].max() - df[indx].min())

        # Save data
        test_df = df.iloc[amount_train_samples:df.shape[0]]
        test_df = shuffle(test_df, random_state=np.random.randint(0, 100))
        df = df[:amount_train_samples]
        df = shuffle(df, random_state=np.random.randint(0, 100))
        test_df.to_csv(formated_test_path, sep=',', index=False)
        df.to_csv(formated_train_path, sep=',', index=False)

        return df

    def format_data_instance(self) -> None:
        """
        Instance method to format the data using instance attributes.
        """
        self.df = DataCls.format_data(self.trainset_path, self.testset_path, self.formated_train_path, self.formated_test_path)
        self.loaded = True


    @staticmethod
    def calculate_obs_size(data_path):
        """
        Calculate the observation size based on the given data path.

        This method calculates the observation size, which is the number of columns in the DataFrame
        minus the number of attack names.

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
        attack_names = []
        if (not os.path.exists(data_path)):
            raise FileNotFoundError(f"File {data_path} not found. Please check the path.")
        
        df = pd.read_csv(data_path, sep=',')
        # Create a list with the existent attacks in the df
        for att in attack_map:
            if att in df.columns:
                # Add only if there exists at least 1
                if np.sum(df[att].values) > 1:
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
                if np.sum(self.df[att].values) > 1 and att not in self.attack_names:
                    self.attack_names.append(att)