import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("Verfügbare GPUs:", gpus)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def log_gpu_info():
    print("\nGPU-CHECK:")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("🚫 GPU not recognized.")
    else:
        print(f"✅ {len(gpus)} recognized GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        for device in devices:
            if device.device_type == 'GPU':
                print(f"  - 💡 Active GPU: {device.name}, Storage: {round(int(device.memory_limit)/1e9,2)} GB")

log_gpu_info()