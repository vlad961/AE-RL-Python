# AE-RL-Python

# Setup Environment
## Windows 11 utilizing GPU
### TensorFlow 
I used miniconda but venv will probably also work out.

1. ```sh
    conda create --name tf-gpu-win python=3.9
   ```
2. Make sure to activate it when running the project (in case you use VSCode, make sure you choose the expected python interpreter)
   ```sh
   conda activate tf-gpu-win 
   ```
3. ```sh
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```

4. ```sh
   pip install --upgrade pip 
   # Alternatively
   python.exe -m pip install --upgrade pip
   ```

5. TensorFlow versions above 2.10 do not support GPU natively on Windows.
   ```sh
   pip install tensorflow<2.11 
   ```

6. ```sh
   pip install numpy<2
   ```

7. If a list of GPU devices is returned by the following command, the TensorFlow installation was successful.
   ```sh
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 
   ```

### Additional libraries
1. ```sh
   conda install pandas # Used for data processing
   ```
   
2. ```sh
   conda install scikit-learn
   ```

3. ```sh
   conda install matplotlib
   ```

## macOS utilizing GPU
### Requirements
- Anaconda/Miniconda
- Tested on macOS 15.1.1 (24B91), Apple M3 Pro
#
1. Create a virtual environment. You can use venv, however I used a miniconda distribution and the following steps.

```sh
conda create --name ae-rl python=3.9
```

2. Work with the project environment
```sh
conda activate ae-rl #activate the env before working with it
conda deactivate # deactivate
```

3. ```sh
    conda install tensorflow
    ```

4. ```sh
    python -m pip install tensorflow-macos
    ```
5. ```sh 
    python -m pip install tensorflow-metal
    ```
6. ```sh 
    conda install pandas
    ```
7. ```sh 
    conda install scikit-learn
    ```
8. ```sh 
    conda install matplotlib
    ```

## Helpful resources to get the project running (Win 11)

 Utilize GPU learning (windows-native):
 1. https://www.tensorflow.org/install/pip#windows-native
 2. https://learn.microsoft.com/de-de/cpp/windows/latest-supported-vc-redist?view=msvc-170 (download newest MS C++ redistributable)
 2. https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing
 3. Anaconda | Miniconda 

# Project Directory Structure
## Core
Contains the whole logic.

## Data
contains plain datasets aswell as the formated one.

## Models
 contains the trained models as well as corresponding test results.


# Get CIC Datasets
To replicate the experiments, download both CIC-Datasets and preprocess them.
I have reused the script of M.Verkeren et. al [Repository](https://gitlab.ilabt.imec.be/mverkerk/ids-dataset-cleaning/-/blob/master/data_cleaning.py?ref_type=heads) with only adjusting the file structure for my repository.

## Download CIC-IDS-2017 Data
Download [CIC-IDS-2017-Data](http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/) unpack GeneratedLabelledFlows.zip.
Save the unpacked directory into `<root>/data/cic-ids/cic-ids-2017/original/`.
## Download CSE-CIC-IDS-2018 Data
1. **Install the AWS CLI**  
   Follow the official guide to install the AWS CLI:  
   [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)

   Example for macOS:
   ```sh
   brew install awscli
   ```

   Example for Windows:
   ```sh
   choco install awscli
   ```

   Example for Ubuntu:
   ```sh
   snap install aws-cli --classic
   ```

   Verify the installation:
   ```sh
   aws --version
   ```

2. **Download the data**  
   Download the data to `<root>/data/datasets/cic-ids/cse-cic-ids-2018/original/`
   Adjust the following example with a corresponding region for your instance:
   ```sh
   mkdir -p data/datasets/cic-ids/cse-cic-ids-2018/original/
   cd data/datasets/cic-ids/cse-cic-ids-2018/original/
   aws s3 sync --no-sign-request --region eu-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" .
   ```

3. **Verify the data**  
   Ensure the data has been successfully downloaded:
   ```sh
   ls data/datasets/cic-ids/cse-cic-ids-2018/original/
   ```

4. **Clean the data** 
 Clean the data by running the[`generate_cleaned_data.py`](./data/datasets/cic-ids/generate_cleaned_data.py) script to preprocess the data. Hint: This might take some time.