import re
import glob
import time
import datetime
from datetime import timedelta
import tarfile

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import SGD
import lightning as L

# add the progress bar
from tqdm import tqdm

from transformers import AdamW
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()


# pre-defined functions
from orgoMolDataLoader import *

def z_denormalize(scaled_labels, labels_mean, labels_std):
    labels = (scaled_labels * labels_std) + labels_mean
    return labels

class omLightning(L.LightningModule):
    def __init__(self, base_model, base_model_output_size, n_classes=1, drop_rate=0.1, pooling= 'cls'):
        super(omLightning, self).__init__()
        D_in, D_out = base_model_output_size, n_classes
        self.model = base_model
        self.dropout = nn.Dropout(drop_rate)
        self.pooling = pooling
        # instantiate a linear regressor
        self.linear_regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )
        

    def forward(self, input_ids, attention_masks):
        hidden_states = self.model(input_ids, attention_masks)
        last_hidden_state = hidden_states.last_hidden_state # [batch_size, input_length, D_in]
        if self.pooling == 'cls':
            input_embedding = last_hidden_state[:,0,:] # [batch_size, D_in] -- [CLS] pooling
        elif self.pooling == 'mean':
            input_embedding = last_hidden_state.mean(dim=1) # [batch_size, D_in] -- mean pooling
        outputs = self.linear_regressor(input_embedding) # [batch_size, D_out]
        return input_embedding, outputs
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3) #learningrate here
        trainDataLoader = self.train_dataloader()
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learningRate,
            epochs=epochs,
            steps_per_epoch=len(trainDataLoader),
            pct_start=0.3,
        )
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        batchInputs, batchMasks, batchLabels, batchNormLabels = batch
        _, predictions = self(batchInputs,batchMasks)
        loss = maeLoss(predictions.squeeze(), batchNormLabels.squeeze())
        return loss
    
    def validation_step(self, batch, batch_idx):
        predictionsList = []
        targetsList = []
        trainLabelsArray = np.array(trainData[property])
        trainLabelsMean = torch.mean(torch.tensor(trainLabelsArray))
        trainLabelsStd = torch.std(torch.tensor(trainLabelsArray))
        batchInputs, batchMasks, batchLabels = batch
        _,predictions = self(batchInputs,batchMasks)
        predictionsDenorm = z_denormalize(predictions, trainLabelsMean, trainLabelsStd)
        predictions = predictionsDenorm.detach().cpu().numpy()
        targets = batchLabels.detach().cpu().numpy()
        for i in range(len(predictions)):
            predictionsList.append(predictions[i][0])
            targetsList.append(targets[i])
        predictionsTensor = torch.tensor(predictionsList)
        targetsTensor = torch.tensor(targetsList)
        val_loss = maeLoss(predictionsTensor.squeeze(),targetsTensor.squeeze()) 
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = nn.L1Loss(y_hat, y)
        self.log("test_loss", test_loss)
    
    def train_dataloader(self):
        return createDataLoaders(
        tokenizer, 
        trainData, 
        maxLength, 
        batchSize, 
        propertyValue=property, 
        pooling=self.pooling, 
        normalize=True, 
        normalizer=normalizerType
        )

    def val_dataloader(self):
        return createDataLoaders(
        tokenizer, 
        validData, 
        maxLength, 
        batchSize, 
        propertyValue=property, 
        pooling=self.pooling
    )

    def test_dataloader(self):
        return createDataLoaders(
        tokenizer, 
        testData, 
        maxLength, 
        batchSize, 
        propertyValue=property, 
        pooling=self.pooling
    )
    
if __name__ == '__main__':
    maeLoss = nn.L1Loss()
    testDataPath = "testSet(1600).csv"
    trainDataPath = "trainingSet(10000).csv"
    validDataPath = "validationSet(400).csv"
    batchSize = 8
    maxLength = 898
    property = 'homoLumoGap'
    normalizerType = 'z_norm'
    learningRate = 1e-3
    epochs = 200

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    baseModel = T5EncoderModel.from_pretrained("t5-v1_1-small",local_files_only= True)
    baseModelOutputSize = 512
    baseModel.resize_token_embeddings(len(tokenizer))

    trainData = pd.read_csv(trainDataPath)
    validData = pd.read_csv(validDataPath)
    testData = pd.read_csv(testDataPath)
    
    model = omLightning(baseModel, baseModelOutputSize)
    trainer = L.Trainer(fast_dev_run = True)
    trainer.fit(model)
