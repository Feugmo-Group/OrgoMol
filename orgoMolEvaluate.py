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
from orgoMolModel import T5Full
from orgoMolDataLoader import *


# for metrics
from torchmetrics.classification import BinaryAUROC

def getRocScore(predictions, targets):
    roc_fn = BinaryAUROC(threshold=None)
    x = torch.tensor(targets)
    y = torch.tensor(predictions)
    y = torch.round(torch.sigmoid(y))
    roc_score = roc_fn(y, x)
    return roc_score

def saveCSV(df, where_to_save):
    df.to_csv(where_to_save, index=False)
    
def timeFormat(total_time):
    """
    Change the from seconds to hh:mm:ss
    """
    total_time_rounded = int(round((total_time)))
    total_time_final = str(datetime.timedelta(seconds=total_time_rounded))
    return total_time_final

def getSequenceLenStats(df, tokenizer, max_len):
    training_on = sum(1 for sent in df['text'].apply(tokenizer.tokenize) if len(sent) <= max_len)
    return (training_on/len(df))*100

def decompressTarCheckpoints(tar_filename):
    tar = tarfile.open(tar_filename)
    tar.extractall()
    tar.close()
    
def evaluate(model, maeLossFunction, testDataLoader, trainLabelsMean, trainLabelsStd,trainLabelsMin,trainLabelsMax, property,device,taskName,normalizer="z_norm"):
    
    testStartTime = time.time()

    model.eval()

    totalTestLoss = 0
    predictionsList = []
    targetsList = []
    
    for step, batch in enumerate(testDataLoader):
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
        
    testPredictions = {f"{property}": predictionsList}

    saveCSV(pd.DataFrame(testPredictions), f"statistics/{taskName}/test_stats_for_{property}.csv")
        
    if taskName == "classification":
        testPerformance = getRocScore(predictionsList, targetsList)
        print(f"\n The roc score achieved on test set for predicting {property} is {testPerformance}")

    elif taskName == "regression":
        predictionsTensor = torch.tensor(predictionsList)
        targetsTensor = torch.tensor(targetsList)
        testPerformance = maeLossFunction(predictionsTensor.squeeze(), targetsTensor.squeeze())
        print(f"\n The mae error achieved on test set for predicting {property} is {testPerformance}")

    averageTestLoss = totalTestLoss / len(testDataLoader)
    testEndingTime = time.time()
    testingTime = timeFormat(testEndingTime-testStartTime)
    print(f"testing took {testingTime} \n")

    return predictionsList, testPerformance

if __name__ == "__main__":
    print("======= Evaluating on test set ========")

    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print("Testing on", torch.cuda.device_count(), "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")


    # set parameters
    batchSize = 8
    maxLength = 512
    dropRate = 0.5
    #preprocessing_strategy = config.get('preprocessing_strategy')
    tokenizerName = 't5_tokenizer'
    pooling = 'cls'
    normalizerType = 'z_norm'
    property = "homoLumoGap"
    taskName = "regression"
    trainDataPath = "trainingSet(5000).csv"
    testDataPath = "testSet(800).csv"
    bestModelPath = "checkpoints/regression/best_checkpoint_for_homoLumoGap.pt"

    # prepare the data
    trainData = pd.read_csv(trainDataPath)
    testData = pd.read_csv(testDataPath)

    # check property type to determine the task name (whether it is regression or classification)
    if testData[property].dtype == 'bool':
        task_name = 'classification'

        #converting True->1.0 and False->0.0
        trainData[property] = trainData[property].astype(float)
        testData[property] = testData[property].astype(float)
    else:
        task_name = 'regression'

    trainLabelsArray = np.array(trainData[property])
    trainLabelsMean = torch.mean(torch.tensor(trainLabelsArray))
    trainLabelsStd = torch.std(torch.tensor(trainLabelsArray))
    trainLabelsMin = torch.min(torch.tensor(trainLabelsArray))
    trainLabelsMax = torch.max(torch.tensor(trainLabelsArray))
    
     # define loss functions
    maeLossFunction = nn.L1Loss()

    freeze = False # a boolean variable to determine if we freeze the pre-trained T5 weights

    # define the tokenizer
    if tokenizerName == 't5_tokenizer': 
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

    elif tokenizerName == 'modified':
        tokenizer = AutoTokenizer.from_pretrained("tokenizers/t5_tokenizer_trained_on_modified_part_of_C4_and_textedge")

    # add defined special tokens to the tokenizer
    if pooling == 'cls':
        tokenizer.add_tokens(["[CLS]"])

    print(f"test data = {len(testData)} samples")
    print('-'*50)
    print(f"testing on {getSequenceLenStats(testData, tokenizer, maxLength)}% samples with whole sequence")
    print('-'*50)

    print("labels statistics on training set:")
    print("Mean:", trainLabelsMean)
    print("Standard deviation:", trainLabelsStd)
    print("Max:", trainLabelsMax)
    print("Min:", trainLabelsMin)
    print("-"*50)

    # define the model
    baseModel = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
    baseModelOutputSize = 512

    # freeze the pre-trained LM's parameters
    if freeze:
        for param in baseModel.parameters():
            param.requires_grad = False

    # resizing the model input embeddings matrix to adapt to newly added tokens by the new tokenizer
    # this is to avoid the "RuntimeError: CUDA error: device-side assert triggered" error
    baseModel.resize_token_embeddings(len(tokenizer))

    # loading the checkpoint of the pretrained model
    
    if "tar.gz" in bestModelPath:
        decompressTarCheckpoints(bestModelPath)
        bestModelPath = bestModelPath[0:-7] + ".pt"

    bestModel = T5Full(baseModel, baseModelOutputSize, drop_rate=dropRate, pooling=pooling)

    deviceIds = [d for d in range(torch.cuda.device_count())]

    if torch.cuda.is_available():
        bestModel = nn.DataParallel(bestModel, device_ids=deviceIds).cuda()

    if isinstance(bestModel, nn.DataParallel):
        bestModel.module.load_state_dict(torch.load(bestModelPath, map_location=torch.device(device)), strict=False)
    else:
        bestModel.load_state_dict(torch.load(bestModelPath, map_location=torch.device(device)), strict=False) 
        bestModel.to(device)

    # create test set dataloaders
    testDataLoader = createDataLoaders(
        tokenizer, 
        testData, 
        maxLength, 
        batchSize, 
        propertyValue=property, 
        pooling=pooling
    )

    _, test_performance = evaluate(bestModel, maeLossFunction, testDataLoader, trainLabelsMean, trainLabelsStd, trainLabelsMin, trainLabelsMax, property, device, task_name, normalizer=normalizerType)
    
    