import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count

def downsample_data(data_arrays, original_sampling_rate, target_sampling_rate):
    # Calculate the downsampling factor
    downsampling_factor = int(original_sampling_rate / target_sampling_rate)

    # Reshape data to have each row represent one channel
    num_channels, num_samples, num_features = data_arrays.shape
    data_arrays_reshaped = data_arrays.reshape(num_channels, num_samples * num_features)

    # Downsample each channel
    downsampled_data = []
    for channel_data in data_arrays_reshaped:
        downsampled_channel = []
        for i in range(0, len(channel_data), downsampling_factor):
            # Take the first sample in each downsampling interval
            downsampled_channel.append(channel_data[i])
        downsampled_data.append(downsampled_channel)

    return np.array(downsampled_data).reshape(num_channels, -1, num_features)

def process_csv_file(file_path, n_rows=250000, n_cols=8):
    # Read the CSV file in chunks
    chunks = pd.read_csv(file_path, chunksize=n_rows)
    dfs = []
    for chunk in chunks:
        # Ensure the data is of the correct shape (250000 x 8)
        if chunk.shape[0] > n_rows:
            chunk = chunk.iloc[:n_rows, :]
        elif chunk.shape[0] < n_rows:
            chunk = chunk.reindex(np.arange(n_rows)).fillna(0)
        
        if chunk.shape[1] > n_cols:
            chunk = chunk.iloc[:, :n_cols]
        elif chunk.shape[1] < n_cols:
            chunk = chunk.reindex(columns=np.arange(n_cols)).fillna(0)
        
        dfs.append(chunk.values)
    return dfs

def process_folder(args):
    folder_path, class_name, n_rows, n_cols = args
    folder_data = []
    folder_label = os.path.basename(folder_path)
    # Iterate over each CSV file in the folder
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                # Process and load the CSV file
                data_arrays = process_csv_file(file_path, n_rows, n_cols)
                folder_data.extend(data_arrays)
    return folder_data, [class_name] * len(folder_data)

def process_mafaulda_data(dataset_dir, n_rows=250000, n_cols=8):
    # Initialize lists to store data arrays and labels
    data_arrays = []
    labels = []

    # Create a pool of processes
    with Pool(processes=cpu_count()) as pool:
        results = []
        # Iterate over each folder in the dataset directory
        for class_id, class_name in enumerate(os.listdir(dataset_dir)):
            class_path = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_path):
                results.append(pool.apply_async(process_folder, [(class_path, class_name, n_rows, n_cols)]))

        for result in results:
            data_arrays_class, labels_class = result.get()
            data_arrays.extend(data_arrays_class)
            labels.extend(labels_class)

    # Convert lists to numpy arrays
    data_arrays = np.array(data_arrays)
    labels = np.array(labels)

    # Assign unique class IDs
    class_ids = np.unique(labels, return_inverse=True)[1]

    return data_arrays, labels, class_ids
