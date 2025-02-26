import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from ztok import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#normalizers were taken from llmprop_utils.py

def z_normalizer(labels):
    """ Implement a z-score normalization technique"""
    labels_mean = np.mean(labels)
    labels_std = np.std(labels)

    scaled_labels = (labels - labels_mean) / labels_std

    return scaled_labels

def z_denormalize(scaled_labels, labels_mean, labels_std):
    labels = (scaled_labels * labels_std) + labels_mean
    return labels

def min_max_scaling(labels):
    """ Implement a min-max normalization technique"""
    min_val = torch.min(labels)
    max_val = torch.max(labels)
    diff = max_val - min_val
    scaled_labels = (labels - min_val) / diff
    return scaled_labels

def mm_denormalize(scaled_labels, min_val, max_val):
    diff = max_val - min_val
    denorm_labels = (scaled_labels * diff) + min_val
    return denorm_labels

def log_scaling(labels):
    """ Implement log-scaling normalization technique"""
    scaled_labels = torch.log1p(labels)
    return scaled_labels

def ls_denormalize(scaled_labels):
    denorm_labels = torch.expm1(scaled_labels)
    return denorm_labels

def createDataLoaders(tokenizer, data, batchSize, property, normalize=True, normalizer=None):
    texts = []
    labels = []
    normalized_labels = []  # Ensure this is initialized
    masks = []
    numerical_values = []

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Creating DataLoaders"):
        zmat_file = row['zmat_file']
        
        # Log the current file being processed
        logging.info(f"Processing file: {zmat_file} (Index: {index})")

        if os.path.exists(zmat_file):
            try:
                tokens, values = tokenizer.tokenize(zmat_file)

                # Append tokens and values directly without NaN checks
                texts.append(tokens)
                numerical_values.append(values)
                labels.append(row[property])  # Collect labels for later normalization

                # Create mask for tokens
                mask = (tokens != 0).astype(int)  # Assuming 0 is the padding token
                # Pad the mask to the longest sequence length
                if len(mask) < tokenizer.longest:
                    mask = np.pad(mask, (0, tokenizer.longest - len(mask)), 'constant', constant_values=0)
                masks.append(mask)

            except Exception as e:
                logging.error(f"Error processing file: {zmat_file} (Index: {index}) - {str(e)}")
        else:
            logging.warning(f"File not found: {zmat_file} (Index: {index})")

    # Check if any valid data was collected
    if texts:
        # Perform normalization after collecting all labels
        if normalize and normalizer is not None:
            if normalizer == 'z_norm':
                normalized_labels = z_normalizer(labels)
            elif normalizer == 'mm_norm':
                normalized_labels = min_max_scaling(labels)
            elif normalizer == 'ls_norm':
                normalized_labels = log_scaling(labels)
            else:
                normalized_labels = labels  # No normalization
        else:
            normalized_labels = labels  # Use original labels if no normalization

        # Create a dataset
        dataset = ZMatDataset(texts, labels, normalized_labels, masks, numerical_values)

        # Create DataLoader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
        return data_loader
    else:
        logging.error("No valid data available after filtering. Please check the input files.")
        return None  # Return None if no valid data is available


