import numpy as np
import os
import pandas as pd

data_dir = "./data/datasets/cse-cic-ids-2018/data/"

files = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
]

no_variance = []

for day, filename in enumerate(files):
    print(f"------- {filename} -------")
    df = pd.read_csv(f"{data_dir}{filename}", skipinitialspace=True)
#     print(df["Label"].value_counts())
#     print(df.columns[df.dtypes == "object"])
    
    print(f"shape: {df.shape}")
    # Drop destination port?  "Dst Port"
    df.drop(columns=["Flow ID", "Src IP", "Src Port", "Dst IP", 
                     'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count',
                       'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
                       'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'], inplace=True, errors="ignore")
    
    # Drop rows with invalid data
    cols=[i for i in df.columns if i not in ["Timestamp", "Label"]]
    for col in cols:
        df[col]=pd.to_numeric(df[col], errors='coerce')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
#     df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"{df.isna().any(axis=1).sum()} rows dropped")
    df.dropna(inplace=True)
    print(f"shape: {df.shape}")
    
    # Drop duplicate rows
    df.drop_duplicates(inplace=True, subset=df.columns.difference(['Label', 'Timestamp']))
    print(f"shape: {df.shape}")
    
    df['Timestamp'] = df['Timestamp'].apply(lambda x: x + pd.Timedelta(hours=12) if x.hour < 8 else x)
    df = df.sort_values(by=['Timestamp'])
    # make clean directory if it doesn't exist
    if not os.path.exists(f"{data_dir}/clean"):
        os.makedirs(f"{data_dir}/clean")

    df[df["Timestamp"] > "2018-01-01"].to_csv(f"{data_dir}/clean/{filename}", index=False)