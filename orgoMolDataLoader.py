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

def tokenize(tokenizer:function, dataframe:pd.DataFrame, maxLength:int):
    """Handles tokenization of your data, prepends CLS tokens and specifies other arguments per LLMProp"""
    encodedCorpus = tokenizer(text=["[CLS] " + str(txt) for txt in dataframe.text.tolist()], 
                                add_special_token = True,
                                padding = 'max_length',
                                truncation = 'longest_first',
                                max_length = maxLength,
                                return_attention_mask = True)
    inputIds = encodedCorpus['input_ids']
    attentionMasks = encodedCorpus['attention_masks']

    return inputIds, attentionMasks


#as of now our training set CSV files does not contain any metric which will need to be added and labelled(VERY IMPORTANT). You would put the pandas column name under property value
def createDataLoaders(tokenizer, dataframe, maxLength, batchSize, propertyValue="TBD", normalize=False, normalizer='z_norm'):
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

    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False) # Set the shuffle to False for now since the labels are continues values check later if this may affect the result

    return dataloader