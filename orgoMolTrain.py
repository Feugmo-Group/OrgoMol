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

import matplotlib.pyplot as plt

# add the progress bar
from tqdm import tqdm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()

# pre-defined functions
from orgoMolModel import T5Encoder
from orgoMolDataLoader import *


# for metrics
from torchmetrics.classification import BinaryAUROC

# set the random seed for reproducibility
torch.manual_seed(50)
np.random.seed(50)

def timeFormat(total_time):
    """
    Change the from seconds to hh:mm:ss
    """
    total_time_rounded = int(round((total_time)))
    total_time_final = str(datetime.timedelta(seconds=total_time_rounded))
    return total_time_final

def getRocScore(predictions, targets):
    roc_fn = BinaryAUROC(threshold=None)
    x = torch.tensor(targets)
    y = torch.tensor(predictions)
    y = torch.round(torch.sigmoid(y))
    roc_score = roc_fn(y, x)
    return roc_score

def compressCheckpointsWithTar(filename):
    filename_for_tar = filename[0:-3]
    tar = tarfile.open(f"{filename_for_tar}.tar.gz", "w:gz")
    tar.add(filename)
    tar.close()

def decompressTarCheckpoints(tar_filename):
    tar = tarfile.open(tar_filename)
    tar.extractall()
    tar.close()

def saveCSV(df, where_to_save):
    df.to_csv(where_to_save, index=False)


def train(model,optimizer,scheduler,bceLossFunction,maeLossFunction,epochs,trainDataLoader,validDataLoader,device,taskName, normalizer = 'z_norm'):
    
    trainingStartingTime = time.time()
    trainingStats = []
    validationPredictions = {}
    
    bestLoss = 1e10 # Set the best loss variable which record the best loss for each epoch
    bestRoc = 0.0
    
    for epoch in range(epochs):
        print(f"========== Epoch {epoch+1}/{epochs} =========")

        epochStartingTime = time.time() 

        totalTrainingLoss = 0
        totalTrainingMaeLoss = 0
        totalTrainingNormalizedMaeLoss = 0
        
        model.train()
        
        for step, batch in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader)):
            
            print(f"Step {step+1}/{len(trainDataLoader)}")
             
            _, predictions = model(batchInputs, batchMasks)
            
            batchInputs, batchMasks, batchLabels, batchNormLabels = tuple(b.to(device) for b in batch)
            
            #the original paper uses different losses for different tasks? Need to figure out how we should handle that
             
            if taskName == 'classification':
                loss = bceLossFunction(predictions.squeeze(), batchLabels.squeeze())
            
            elif taskName == 'regression':
                loss = maeLossFunction(predictions.squeeze(), batchNormLabels.squeeze())
                
                if normalizer == 'z_norm':
                    predictionsDenorm = z_denormalize(predictions, trainLabelMean, trainLabelsStd)

                elif normalizer == 'mm_norm':
                    predictionsDenorm = mm_denormalize(predictions, trainLabelsMin, trainLabelsMax)

                elif normalizer == 'ls_norm':
                    predictionsDenorm = ls_denormalize(predictions)

                elif normalizer == 'no_norm':
                    loss = maeLossFunction(predictions.squeeze(), batchLabels.squeeze())
                    predictionsDenorm = predictions

                maeLoss = maeLossFunction(predictionsDenorm.squeeze(), batchLabels.squeeze()) 
            
            if taskName == "classification":
                totalTrainingLoss += loss.item()
            
            elif taskName == "regression":
                totalTrainingLoss += maeLoss.item()
            
            # back propagate
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # average training loss on actual output
        averageTrainingLoss = totalTrainingLoss/len(trainDataLoader) 
        
        epochEndingTime = time.time()
        trainingTime = timeFormat(epochEndingTime - epochStartingTime)

        print(f"Average training loss = {averageTrainingLoss}")
        print(f"Training for this epoch took {trainingTime}")

        # Validation
        print("")
        print("Running Validation ....")

        validStartTime = time.time()

        model.eval()
        
        totalEvalMaeLoss = 0
        predictionsList = []
        targetsList = []
        
        for step, batch in tqdm(enumerate(validDataLoader), total=len(validDataLoader)):
            batchInputs, batchMasks, batchLabels = tuple(b.to(device) for b in batch)

            with torch.no_grad():
                _, predictions = model(batchInputs, batchMasks)

                if taskName == "classification":
                    predictionsDenorm = predictions

                elif taskName == "regression":
                    
                    if normalizer == 'z_norm':
                        predictionsDenorm = z_denormalize(predictions, trainLabelsMean, trainLabelsStd)

                    elif normalizer == 'mm_norm':
                        predictionsDenorm = mm_denormalize(predictions, trainLabelsMin, trainLabelsMax)

                    elif normalizer == 'ls_norm':
                        predictionsDenorm = ls_denormalize(predictions)

                    elif normalizer == 'no_norm':
                        predictionsDenorm = predictions

            predictions = predictionsDenorm.detach().cpu().numpy()
            targets = batchLabels.detach().cpu().numpy()
            
            for i in range(len(predictions)):
                predictionsList.append(predictions[i][0])
                targetsList.append(targets[i])
            
            validEndingTime = time.time()
            validationTime = timeFormat(validEndingTime-validStartTime)
            
            
        if taskName == "classification":
            validPerformance = getRocScore(predictionsList, targetsList)
            
            if validPerformance >= bestRoc:
                bestRoc = validPerformance
                bestEpoch = epoch+1

                # save the best model checkpoint
                save_to_path = f"checkpoints/{taskName}/best_checkpoint_for_{property}.pt"
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), save_to_path)
                    compressCheckpointsWithTar(save_to_path)
                else:
                    torch.save(model.state_dict(), save_to_path)
                    compressCheckpointsWithTar(save_to_path)
                
                # save statistics of the best model
                trainingStats.append(
                    {
                        "best_epoch": epoch + 1,
                        "training_loss": averageTrainingLoss,
                        "validation_roc_score": validPerformance,
                        "training time": trainingTime,
                        "validation time": validationTime
                    }
                )

                validationPredictions.update(
                    {
                        f"epoch_{epoch+1}": predictionsList
                    }
                )

                saveCSV(pd.DataFrame(data=trainingStats), f"statistics/{taskName}/training_stats_for_{property}.csv")
                saveCSV(pd.DataFrame(validationPredictions), f"statistics/{taskName}/validation_stats_for_{property}.csv")

            else:
                bestRoc = bestRoc

            print(f"Validation roc score = {validPerformance}")
            
        elif taskName == "regression":
            predictionsTensor = torch.tensor(predictionsList)
            targetsTensor = torch.tensor(targetsList)
            validPerformance = maeLossFunction(predictionsTensor.squeeze(), targetsTensor.squeeze())
        
            if validPerformance <= bestLoss:
                bestLoss = validPerformance
                bestEpoch = epoch+1
                
                # save the best model checkpoint
                save_to_path = f"checkpoints/{taskName}/best_checkpoint_for_{property}.pt"
                
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), save_to_path)
                    compressCheckpointsWithTar(save_to_path)
                else:
                    torch.save(model.state_dict(), save_to_path)
                    compressCheckpointsWithTar(save_to_path)
                
                # save statistics of the best model
                trainingStats.append(
                    {
                        "best_epoch": epoch + 1,
                        "training mae loss": averageTrainingLoss,
                        "validation mae loss": validPerformance,
                        "training time": trainingTime,
                        "validation time": validationTime
                    }
                )

                validationPredictions.update(
                    {
                        f"epoch_{epoch+1}": predictionsList
                    }
                )

                saveCSV(pd.DataFrame(data=trainingStats), f"statistics/{taskName}/training_stats_for_{property}.csv")
                saveCSV(pd.DataFrame(validationPredictions), f"statistics/{taskName}/validation_stats_for_{property}.csv")

            else:
                bestLoss = bestLoss
            
            print(f"Validation mae error = {validPerformance}")
            
        print(f"validation took {validationTime}")
        
    trainEndingTime = time.time()
    totalTrainingTime = trainEndingTime-trainingStartingTime

    print("\n========== Training complete ========")
    print(f"Training LLM_Prop on {property} prediction took {timeFormat(totalTrainingTime)}")

    if taskName == "classification":
        print(f"The lowest roc score achieved on validation set on {property} is {bestRoc} at {bestEpoch}th epoch \n")

    elif taskName == "regression":
        print(f"The lowest mae error achieved on validation set on predicting {property} is {bestLoss} at {bestEpoch}th epoch \n")
    
    return trainingStats, validationPredictions

def evaluate( model, maeLossFunction, testDataLoader, trainLabelsMean, trainLabelsStd,trainLabelsMin,trainLabelsMax, property,device,taskName,normalizer="z_norm"):
    
    
    