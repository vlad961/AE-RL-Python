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

5. ```sh
   pip install tensorflow<2.11 # Anything above 2.10 is not supported on the GPU on Windows
   ```

6. ```sh
   pip install numpy<2
   ```

7. ```sh
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" # If a list of GPU devices is returned the TensorFlow installation was successful.
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


## Helpful resources to get the project running.

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
