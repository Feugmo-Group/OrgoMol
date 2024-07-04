import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


#normalizers were taken from llmprop_utils.py

def z_normalizer(labels):
    """ Implement a z-score normalization technique"""
    labels_mean = torch.mean(labels)
    labels_std = torch.std(labels)

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

def tokenize(tokenizer:callable, dataframe:pd.DataFrame, maxLength:int, pooling = 'cls'):
    """
    1. Takes in the the list of input sequences and return 
    the input_ids and attention masks of the tokenized sequences
    2. max_length = the max length of each input sequence 
    """
    if pooling == 'cls':
        encoded_corpus = tokenizer(text=["[CLS] " + str(descr) for descr in dataframe.text.tolist()],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation=True,
                                    max_length= maxLength, # According to ByT5 paper
                                    return_attention_mask=True)
    elif pooling == 'mean':
        encoded_corpus = tokenizer(text=dataframe.text.tolist(),
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation=True,
                                    max_length= maxLength, # According to ByT5 paper
                                    return_attention_mask=True) 
        
    input_ids = encoded_corpus['input_ids']
    attention_masks = encoded_corpus['attention_mask']

    return input_ids, attention_masks

#as of now our training set CSV files does not contain any metric which will need to be added and labelled(VERY IMPORTANT). You would put the pandas column name under property value
def createDataLoaders(tokenizer, dataframe, maxLength, batchSize, propertyValue= 'homoLumoGap', normalize=False, normalizer='z_norm',pooling='cls'):
    """
    Dataloader which arrange the input sequences, attention masks, and labels in batches
    and transform the to tensors
    """
    input_ids, attention_masks = tokenize(tokenizer, dataframe, maxLength)
    labels = dataframe[propertyValue].to_numpy()
    
    inputTensor = torch.tensor(input_ids)
    maskTensor = torch.tensor(attention_masks)
    labelsTensor = torch.tensor(labels)
    
    if normalize:
        if normalizer == 'z_norm':
            normalizedLabels = z_normalizer(labelsTensor)
        elif normalizer == 'mm_norm':
           normalizedLabels = min_max_scaling(labelsTensor)
        elif normalizer == 'ls_norm':
            normalizedLabels = log_scaling(labelsTensor)
        elif normalizer == 'no_norm':
            normalizedLabels = labelsTensor

        dataset = TensorDataset(inputTensor, maskTensor, labelsTensor, normalizedLabels)
    else:
        dataset = TensorDataset(inputTensor, maskTensor, labelsTensor)

    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False) # Set the shuffle to False for now since the labels are continious values check later if this may affect the result

    return dataloader
