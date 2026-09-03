import numpy as np
import time
import gc
import psutil
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import matplotlib.pyplot as plt

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.Process().memory_info().rss / (1024**3)

def reduce_dataset(data_save, num_time_to, num_node_red, num_param, num_time, num_node_red_start, num_node_red_end):

    start = time.time()

    num_node = data_save.shape[-1]
    if num_time_to == num_time and num_node_red == num_node:
        FOM_data = data_save
    else:
        FOM_data_temp = data_save
        num_time = num_time_to
        FOM_data = np.zeros((num_param, num_time, num_node_red))
        FOM_data[:, 0:num_time,:] = FOM_data_temp[:, 0:num_time, num_node_red_start:num_node_red_end]
        del FOM_data_temp

        num_node = num_node_red
        FOM_data_temp = np.zeros((num_param, num_time, num_node_red))
        FOM_data_temp[:,:,0:num_node] = FOM_data
        FOM_data_temp[:,:,num_node:-1] = 0
        del FOM_data

        FOM_data = np.zeros((num_param, num_time, num_node))
        FOM_data = FOM_data_temp

    end = time.time()
    print();print()
    print(f"Dataset reduction completed in {end - start:.2f}s")
    print(f"Dataset reduced to FOM data shape: {FOM_data.shape}")

    return num_time, FOM_data, num_node

def data_augmentation(stretch, FOM_data, num_param, num_node):
    if stretch == 1:
        new_x_train = FOM_data

        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.05, p=1),
            Resample(min_sample_rate=1000, max_sample_rate=15000, p=1),
            Shift(p=1),
        ])

        for i in range(num_param):
            X = FOM_data[i, :, :]
            X = augment(samples=X, sample_rate=10000)
            new_x_train[i, :, 0:num_node] = X[:, 0:num_node]

        FOM_data_aug = np.append(FOM_data, new_x_train, axis=0)
    else:
        FOM_data_aug = FOM_data

    return FOM_data_aug


def data_scaler(FOM_data_aug, FOM_data, num_time, num_node, directory, chunk_size=None):
    start = time.time()
    initial_memory = get_memory_usage()
    
    if chunk_size is None:
        chunk_size = 10000

    
    print();print()
    print(f"Fitting scaler on dataset of shape: {FOM_data_aug.shape}")
    
    # if FOM_data_aug.dtype != np.float32:
    #     print("Converting to float32...")
    #     FOM_data_aug = FOM_data_aug.astype(np.float32)
    #     gc.collect()
        
    # if FOM_data.dtype != np.float32:
    #     FOM_data = FOM_data.astype(np.float32)
    #     gc.collect()
    
    # after_conversion_memory = get_memory_usage()
    # conversion_change = after_conversion_memory - initial_memory
    # print(f"Memory after float32 conversion: {after_conversion_memory:.2f} GB ({conversion_change:+.2f} GB)")
    
    # Wider range for better VAE training - VAEs work better with [-1, 1] or [-0.9, 0.9]
    scaler = MinMaxScaler(feature_range=(-0.7, 0.7))
    
    # For MinMaxScaler, we need to fit on a representative sample since it doesn't support partial_fit
    total_samples = FOM_data_aug.shape[0] * FOM_data_aug.shape[1]
    
    # Simplified and efficient sampling - use every Nth sample
    max_samples = min(50000, total_samples // 10)  # Use 10% of data or 50k samples
    if max_samples < 1000:
        max_samples = min(1000, total_samples)  # Minimum 1000 samples
    
    sample_stride = max(1, total_samples // max_samples)
    print(f"Sampling {max_samples} representative samples (every {sample_stride}th sample)")
    
    # Simple random sampling approach - much more efficient
    np.random.seed(42)  # Reproducible sampling
    if total_samples > max_samples:
        sample_indices = np.random.choice(total_samples, max_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
    
    # Convert to coordinates and extract samples efficiently
    param_indices = sample_indices // num_time
    time_indices = sample_indices % num_time
    
    representative_samples = FOM_data_aug[param_indices, time_indices, :]

    scaler.fit(representative_samples)
    del representative_samples

    # Transform data in chunks to avoid memory issues
    print("Transforming training data in chunks...")
    FOM_data_aug_flat = FOM_data_aug.reshape(-1, num_node)
    total_samples = FOM_data_aug_flat.shape[0]
    
    # More reasonable chunk size for progress reporting
    if total_samples > 100000:
        report_interval = chunk_size * 5  # Report every 5 chunks for large datasets
    else:
        report_interval = chunk_size  # Report every chunk for smaller datasets
    
    chunks_processed = 0
    total_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    # Process in chunks to avoid memory issues
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        FOM_data_aug_flat[start_idx:end_idx] = scaler.transform(FOM_data_aug_flat[start_idx:end_idx])
        chunks_processed += 1
        
        # Clean progress reporting
        if start_idx % report_interval == 0 or end_idx == total_samples:
            progress = (end_idx / total_samples) * 100
            print(f"  Progress: {progress:.1f}% ({chunks_processed}/{total_chunks} chunks)")
    
    # Reshape back in-place
    new_x_train = FOM_data_aug_flat.reshape(FOM_data_aug.shape)
    DATA_shape = new_x_train.shape[1:]
    
    # Clean up intermediate variables
    del FOM_data_aug_flat
    gc.collect()
    
    # Save scaler
    dump(scaler, open('./model_save/scaler.pkl', 'wb'))
    
    # Final summary
    final_memory = get_memory_usage()
    end = time.time()
    memory_change = final_memory - initial_memory
    
    print(f"   Data scaling completed in {end - start:.2f} seconds")
    print(f"   Final data shape: {new_x_train.shape}, dtype: {new_x_train.dtype}")
    print(f"   Memory usage: {initial_memory:.2f} GB → {final_memory:.2f} GB ({memory_change:+.2f} GB)")
    print(f"   Scaler saved to: ./model_save/scaler.pkl")
    
    return new_x_train, DATA_shape, scaler

def latent_conditioner_scaler(data, name):
    # Also update LatentConditioner scaler to match
    scaler = MinMaxScaler(feature_range=(-0.7, 0.7))

    # Handle 3D arrays by reshaping to 2D for scaling
    original_shape = data.shape
    
    # Check for empty data
    if original_shape[0] == 0:
        raise ValueError(f"Empty data array detected with shape {original_shape}. "
                        "Please check your data loading configuration. "
                        "If using 'input_type image', ensure PNG files exist in the specified directory.")
    
    if len(original_shape) == 3:
        # Reshape to 2D for scaler
        data_reshaped = data.reshape(original_shape[0], -1)
    else:
        data_reshaped = data

    scaler.fit(data_reshaped)
    scaled_data = scaler.transform(data_reshaped)

    # Reshape back to original dimensions
    if len(original_shape) == 3:
        scaled_data = scaled_data.reshape(original_shape)

    dump(scaler, open(name, 'wb'))

    return scaled_data, scaler
