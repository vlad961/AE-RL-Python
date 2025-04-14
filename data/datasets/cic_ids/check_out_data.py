import os
import pandas as pd

def check_out_data():
    """
    Check out the data from the datasets/cic-ids directory.
    """
    # current path
    current_path = os.path.dirname(os.path.abspath(__file__))
    cic_2017_path = os.path.join(current_path, 'cic-ids-2017/clean/all_data.feather')
    cic_2018_path = os.path.join(current_path, 'cse-cic-ids-2018/clean/all_data.feather')
    # Load the data
    cic_2017 = pd.read_feather(cic_2017_path)
    cic_2018 = pd.read_feather(cic_2018_path)
    # Display the first few rows of the data
    print(cic_2017.head())
    print(cic_2018.head())
    # Display the shape of the data
    print(f'cic_2017 shape: {cic_2017.shape}')
    print(f'cic_2018 shape: {cic_2018.shape}')
    # Display the columns of the data
    print(f'cic_2017 columns: {cic_2017.columns}')
    print(f'cic_2018 columns: {cic_2018.columns}')
    # Display the data types of the data
    print(f'cic_2017 dtypes: {cic_2017.dtypes}')
    print(f'cic_2018 dtypes: {cic_2018.dtypes}')
    # Display the memory usage of the data
    print(f'cic_2017 memory usage: {cic_2017.memory_usage(deep=True)}')
    print(f'cic_2018 memory usage: {cic_2018.memory_usage(deep=True)}')
    # Display the null values of the data
    print(f'cic_2017 null values: {cic_2017.isnull().sum()}')
    print(f'cic_2018 null values: {cic_2018.isnull().sum()}')
    # Display the unique values of the data
    print(f'cic_2017 unique values: {cic_2017.nunique()}')
    print(f'cic_2018 unique values: {cic_2018.nunique()}')

    print('Data check out completed.')

check_out_data()