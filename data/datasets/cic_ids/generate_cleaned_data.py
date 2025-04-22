import pandas as pd
import numpy as np
import random 
import os
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(0)

drop_columns = [
    # Dataset Specific Information
    "Flow ID", 
    "Source IP", "Src IP", 
    "Source Port", "Src Port", 
    "Destination IP", "Dst IP",
    # Features Without Observed Variance
    "Bwd PSH Flags", 
    "Fwd URG Flags", 
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Byts/b Avg", 
    "Fwd Avg Packets/Bulk", "Fwd Pkts/b Avg", 
    "Fwd Avg Bulk Rate", "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk", "Bwd Byts/b Avg", 
    "Bwd Avg Packets/Bulk", "Bwd Pkts/b Avg", 
    "Bwd Avg Bulk Rate", "Bwd Blk Rate Avg",
    # Duplicate Column
    'Fwd Header Length.1'
]

mapper = {
    'Dst Port': 'Destination Port',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Fwd Packets Length Total', # 18 ->
    'Total Length of Fwd Packets': 'Fwd Packets Length Total', # 17 ->
    'TotLen Bwd Pkts': 'Bwd Packets Length Total', # 18 ->
    'Total Length of Bwd Packets': 'Bwd Packets Length Total', # 17 ->
    'Fwd Pkt Len Max': 'Fwd Packet Length Max', # 18 -> 17
    'Fwd Pkt Len Min': 'Fwd Packet Length Min', # 18 -> 17
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean', # 18 -> 17
    'Fwd Pkt Len Std': 'Fwd Packet Length Std', # 18 -> 17
    'Bwd Pkt Len Max': 'Bwd Packet Length Max', # 18 -> 17
    'Bwd Pkt Len Min': 'Bwd Packet Length Min', # 18 -> 17
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean', # 18 -> 17
    'Bwd Pkt Len Std': 'Bwd Packet Length Std', # 18 -> 17
    'Flow Byts/s': 'Flow Bytes/s', # 18 -> 17
    'Flow Pkts/s': 'Flow Packets/s', # 18 -> 17
    'Fwd IAT Tot': 'Fwd IAT Total', # 18 -> 17
    'Bwd IAT Tot': 'Bwd IAT Total', # 18 -> 17
    'Fwd Header Len': 'Fwd Header Length', # 18 -> 17
    'Bwd Header Len': 'Bwd Header Length', # 18 -> 17
    'Fwd Pkts/s': 'Fwd Packets/s', # 18 -> 17
    'Bwd Pkts/s': 'Bwd Packets/s', # 18 -> 17
    'Pkt Len Min': 'Packet Length Min', # 18
    'Min Packet Length': 'Packet Length Min', # 17 
    'Pkt Len Max': 'Packet Length Max', # 18
    'Max Packet Length': 'Packet Length Max', # 17
    'Pkt Len Mean': 'Packet Length Mean', # 18 -> 17
    'Pkt Len Std': 'Packet Length Std', #  18 -> 17
    'Pkt Len Var': 'Packet Length Variance', # 18 -> 17
    'FIN Flag Cnt': 'FIN Flag Count', # 18 -> 17
    'SYN Flag Cnt': 'SYN Flag Count', # 18 -> 17
    'RST Flag Cnt': 'RST Flag Count', # 18 -> 17
    'PSH Flag Cnt': 'PSH Flag Count', # 18 -> 17
    'ACK Flag Cnt': 'ACK Flag Count', # 18 -> 17
    'URG Flag Cnt': 'URG Flag Count', # 18 -> 17
    'ECE Flag Cnt': 'ECE Flag Count', # 18 -> 17
    'Pkt Size Avg': 'Avg Packet Size', # 18 
    'Average Packet Size': 'Avg Packet Size', # 17
    'Fwd Seg Size Avg': 'Avg Fwd Segment Size', # 18 -> 17
    'Bwd Seg Size Avg': 'Avg Bwd Segment Size', # 18 -> 17
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk', # 18 -> 17 will be dropped
    'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk', # 18 -> 17 will be dropped
    'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate', # 18 -> 17 will be dropped
    'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk', # 18 -> 17 will be dropped
    'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk', # 18 -> 17 will be dropped
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate', # 18 -> 17 will be dropped
    'Subflow Fwd Pkts': 'Subflow Fwd Packets', # 18 -> 17
    'Subflow Fwd Byts': 'Subflow Fwd Bytes', # 18 -> 17
    'Subflow Bwd Pkts': 'Subflow Bwd Packets', # 18 -> 17
    'Subflow Bwd Byts': 'Subflow Bwd Bytes', # 18 -> 17
    'Init Fwd Win Byts': 'Init Fwd Win Bytes', # 18
    'Init_Win_bytes_forward': 'Init Fwd Win Bytes', # 17
    'Init Bwd Win Byts': 'Init Bwd Win Bytes', # 18
    'Init_Win_bytes_backward': 'Init Bwd Win Bytes', # 17
    'Fwd Act Data Pkts': 'Fwd Act Data Packets', # 18
    'act_data_pkt_fwd': 'Fwd Act Data Packets', # 17
    'Fwd Seg Size Min': 'Fwd Seg Size Min', # 18 -> 18
    'min_seg_size_forward': 'Fwd Seg Size Min' # 17 -> 18
}

def clean_dataset(dataset, filetypes=['feather']):
    # Will search for all files in the dataset subdirectory 'orignal'
    for file in os.listdir(f'{dataset}/original'):
        if not file.endswith('.csv'):
            print(f"Ignore file: {file}")
            continue
        print(f"------- {file} -------")
        df = pd.read_csv(f"{dataset}/original/{file}", skipinitialspace=True, encoding='latin', low_memory=False)
        print(df["Label"].value_counts())
        print(f"Shape before transformation: {df.shape}")

        # Rename column names for uniform column names across files
        df.rename(columns=mapper, inplace=True)

        # I. Drop unrelevant columns
        df.drop(columns=drop_columns, inplace=True, errors="ignore")

        # Parse Timestamp column to pandas datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x + pd.Timedelta(hours=12) if x.hour < 8 else x)
        df = df.sort_values(by=['Timestamp'])

        # Make Label column Categorical
        df['Label'].replace({'BENIGN': 'Benign'}, inplace=True)
        df['Label'] = df.Label.astype('category')

        # Parse Columns to correct dtype
        int_col = df.select_dtypes(include='integer').columns
        df[int_col] = df[int_col].apply(pd.to_numeric, errors='coerce', downcast='integer')
        float_col = df.select_dtypes(include='float').columns
        df[float_col] = df[float_col].apply(pd.to_numeric, errors='coerce', downcast='float')
        obj_col = df.select_dtypes(include='object').columns
        print(f'Columns with dtype == object: {obj_col}')
        df[obj_col] = df[obj_col].apply(pd.to_numeric, errors='coerce')

        # II. Drop rows with invalid data
        print(f"Shape before replacing np.inf values to np.nan: {df.shape}")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"{df.isna().any(axis=1).sum()} invalid rows dropped")
        df.dropna(inplace=True)

        # III. Drop duplicate rows
        print(f"Shape before dropping duplicates: {df.shape}")
        print(f"Value counts before dropping duplicates: {df['Label'].value_counts()}")
        df.drop_duplicates(inplace=True, subset=df.columns.difference(['Label', 'Timestamp']))
        print(f"Value counts after dropping duplicates: {df['Label'].value_counts()}")
        print(f"Shape after dropping duplicates: {df.shape}\n")

        # Reset index
        df.reset_index(inplace=True, drop=True)

        # Plot resulting file
        #plot_day(df)
        # if clean directory does not exist, create it
        if not os.path.exists(f'{dataset}/clean'):
            os.makedirs(f'{dataset}/clean')

        # Save to file
        if 'feather' in filetypes:
            df.to_feather(f'{dataset}/clean/{file}.feather')
        if 'parquet' in filetypes:
            df.to_parquet(f'{dataset}/clean/{file}.parquet', index=False)
        # Save additional to .csv for easy access
        if 'csv' in filetypes:
            df.to_csv(f'{dataset}/clean/{file}', index=False)
            
def aggregate_data(dataset, save=True, filetype='feather'):
    # Will search for all files in the 'clean' directory of the correct filetype and aggregate them
    all_data = pd.DataFrame()
    for file in glob.glob(f'{dataset}/clean/*.{filetype}'):
        print(file)
        df = pd.DataFrame()
        if filetype == 'feather':
            df = pd.read_feather(file)
        if filetype == 'parquet':
            df = pd.read_parquet(file)
        if filetype == 'csv':
            df = pd.read_csv(file)
        print(df.shape)
        print(f'{df["Label"].value_counts()}\n')
        all_data = pd.concat([all_data, df], ignore_index=True)
    print('ALL DATA')
    duplicates = all_data[all_data.duplicated(subset=all_data.columns.difference(['Label', 'Timestamp']))]
    print('Removed duplicates after aggregating:')
    print(duplicates.Label.value_counts())
    print('Resulting Dataset')
    all_data.drop(duplicates.index, axis=0, inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    print(all_data.shape)
    print(f'{all_data["Label"].value_counts()}\n')
    if save:
        malicious = all_data[all_data.Label != 'Benign'].reset_index(drop = True)
        benign = all_data[all_data.Label == 'Benign'].reset_index(drop = True)
        if filetype == 'feather':
            all_data.to_feather(f'{dataset}/clean/all_data.feather')
            malicious.to_feather(f'{dataset}/clean/all_malicious.feather')
            benign.to_feather(f'{dataset}/clean/all_benign.feather')
        if filetype == 'parquet':
            all_data.to_parquet(f'{dataset}/clean/all_data.parquet', index=False)
            malicious.to_parquet(f'{dataset}/clean/all_malicious.parquet', index=False)
            benign.to_parquet(f'{dataset}/clean/all_benign.parquet', index=False)
        if filetype == 'csv':
            all_data.to_csv(f'{dataset}/clean/all_data.csv', index=False)
            malicious.to_csv(f'{dataset}/clean/all_malicious.csv', index=False)
            benign.to_csv(f'{dataset}/clean/all_benign.csv', index=False)
            
if __name__ == "__main__":
    # Adjust for cleaning the correct dataset into the desired format
    
    # Needs directory with dataset name containing dir 'original' containing de csv's
    cic_ids_2017 = 'cic-ids-2017'
    filetypes = ['feather', 'csv'] # Supported types ['feather', 'parquet', 'csv']
    clean_dataset(cic_ids_2017, filetypes=filetypes)
    aggregate_data(cic_ids_2017, save=True, filetype='feather')
    aggregate_data(cic_ids_2017, save=True, filetype='csv')

    cse_cic_ids_2018 = 'cse-cic-ids-2018'
    clean_dataset(cse_cic_ids_2018, filetypes=filetypes)
    aggregate_data(cse_cic_ids_2018, save=True, filetype='feather')
    aggregate_data(cse_cic_ids_2018, save=True, filetype='csv')