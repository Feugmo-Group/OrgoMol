
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint


from transformers import AutoTokenizer, T5EncoderModel
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()


# pre-defined functions
from orgoMolDataLoader import *
from preProcess import *

def z_denormalize(scaled_labels, labels_mean, labels_std):
    labels = (scaled_labels * labels_std) + labels_mean
    return labels

def saveCSV(df, where_to_save):
    df.to_csv(where_to_save, index=False)
    

class omLightning(L.LightningModule):
    def __init__(self, base_model, base_model_output_size, n_classes=1, drop_rate=0.1,pooling="cls"):
        super().__init__()
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=learningRate) 
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learningRate,
            epochs=epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
        )
        return ([optimizer], [lr_scheduler])
    
    def training_step(self, batch, batch_idx):
        batchInputs, batchMasks, batchLabels, batchNormLabels = tuple(b for b in batch)
        _, predictions = self(batchInputs,batchMasks)
        loss = maeLoss(predictions.squeeze(), batchNormLabels.squeeze())
        self.log("trainLoss", loss.to('cuda'), prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        predictionsList = []
        targetsList = []
        trainingStats = []
        trainLabelsArray = np.array(trainData[property])
        trainLabelsMean = torch.mean(torch.tensor(trainLabelsArray))
        trainLabelsStd = torch.std(torch.tensor(trainLabelsArray))
        batchInputs, batchMasks, batchLabels = tuple(b for b in batch)
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
        self.log("valLoss", val_loss.to('cuda'), prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
            
    def test_step(self, batch, batch_idx):
        predictionsList = []
        targetsList = []
        batchInputs, batchMasks, batchLabels = tuple(b for b in batch)
        _, predictions = self(batchInputs,batchMasks)
        trainLabelsArray = np.array(trainData[property])
        trainLabelsMean = torch.mean(torch.tensor(trainLabelsArray))
        trainLabelsStd = torch.std(torch.tensor(trainLabelsArray))
        predictionsDenorm = z_denormalize(predictions, trainLabelsMean, trainLabelsStd)
        predictions = predictionsDenorm.detach().cpu().numpy()
        targets = batchLabels.detach().cpu().numpy()
        for i in range(len(predictions)):
            predictionsList.append(predictions[i][0])
            targetsList.append(targets[i])
        testPredictions = {f"{property}": predictionsList}
        saveCSV(pd.DataFrame(testPredictions), f"testStatsFor{experimentName}.csv")    
        predictionsTensor = torch.tensor(predictionsList)
        targetsTensor = torch.tensor(targetsList)
        test_loss = maeLoss(predictionsTensor.squeeze(),targetsTensor.squeeze())
        self.log("test_loss", test_loss.to('cuda'),sync_dist=True)
        
    
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
    #Hyperparameters
    testDataPath = "testSet(800).csv"
    trainDataPath = "trainingSet(5000).csv"
    validDataPath = "validationSet(200).csv"
    batchSize = 8
    maxLength = 888
    property = 'homoLumoGap'
    normalizerType = 'z_norm'
    learningRate = 1E-3
    epochs = 200
    dropRate=0.5
    preProcessingStrategy = "None"
    pooling = "mean"
    loggerPath = "OrgoMol5K"
    experimentName = "5KBase"
    
    tokenizer = AutoTokenizer.from_pretrained("t5-v1_1-small",local_files_only = True)
    baseModel = T5EncoderModel.from_pretrained("t5-v1_1-small",local_files_only= True)
    if pooling == 'cls':
        tokenizer.add_tokens(["[CLS]"])
    if preProcessingStrategy == 'Both' or "Token":
        tokenizer.add_tokens('[ANG]')
        tokenizer.add_tokens('[NUM]')
    
    baseModelOutputSize = 512
    baseModel.resize_token_embeddings(len(tokenizer))
    pad_to_multiple_of = 8
    
    

    #checkpointing
    checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="valLoss",
    mode="min",
    dirpath=f"{loggerPath}/{experimentName}",
    filename=experimentName+ " -{epoch:02d}-{val_loss:.2f}")
    
    
    trainData = preProcess(pd.read_csv(trainDataPath),preProcessingStrategy)
    validData = preProcess(pd.read_csv(validDataPath),preProcessingStrategy)
    testData = preProcess(pd.read_csv(testDataPath),preProcessingStrategy)
   
    model = omLightning(baseModel, baseModelOutputSize,drop_rate=dropRate,pooling=pooling)
    logger = TensorBoardLogger(loggerPath, name=experimentName)
    seed_everything(50, workers=True)
    trainer = L.Trainer(logger=logger,deterministic=True,devices=1,num_nodes=1,accelerator='auto',max_epochs=epochs,callbacks=[checkpoint_callback])
    trainer.fit(model)
    trainer.test(ckpt_path="best")
