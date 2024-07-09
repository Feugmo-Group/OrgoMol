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
from preProcess import *

# for metrics
from torchmetrics.classification import BinaryAUROC

from lightning.fabric import Fabric

# set the random seed for reproducibility
torch.manual_seed(50)
np.random.seed(50)
fabric = Fabric()
fabric.launch()

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
    
def getSequenceLenStats(df, tokenizer, max_len):
    training_on = sum(1 for sent in df['text'].apply(tokenizer.tokenize) if len(sent) <= max_len)
    return (training_on/len(df))*100


def train(model,optimizer,scheduler,bceLossFunction,maeLossFunction,epochs,trainDataLoader,validDataLoader,taskName='regression', normalizer = 'z_norm'):
    
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

            batchInputs, batchMasks, batchLabels, batchNormLabels = tuple(b for b in batch)

            _, predictions = model(batchInputs, batchMasks)
             
            if taskName == 'classification':
                loss = bceLossFunction(predictions.squeeze(), batchLabels.squeeze())
            
            elif taskName == 'regression':
                loss = maeLossFunction(predictions.squeeze(), batchNormLabels.squeeze())
                
                if normalizer == 'z_norm':
                    predictionsDenorm = z_denormalize(predictions, trainLabelsMean, trainLabelsStd)

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
            fabric.backward(loss)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # average training loss on actual output
        averageTrainingLoss = totalTrainingLoss/len(trainDataLoader) 
        
        epochEndingTime = time.time()
        trainingTime = timeFormat(epochEndingTime - epochStartingTime)

        print(f'Average training loss = {averageTrainingLoss}')
        print(f'Training for this epoch took {trainingTime}')

        print("")
        print("Running Validation ...")

        validStartTime = time.time()

        model.eval()
        
        totalEvalMaeLoss = 0
        predictionsList = []
        targetsList = []
        
        for step, batch in tqdm(enumerate(validDataLoader), total=len(validDataLoader)):
            batchInputs, batchMasks, batchLabels = tuple(b for b in batch)

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
                save_to_path = f"checkpoints/{experimentName}/best_checkpoint_for_{property}.pt"
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

                saveCSV(pd.DataFrame(data=trainingStats), f"statistics/{experimentName}/training_stats_for_{property}.csv")
                saveCSV(pd.DataFrame(validationPredictions), f"statistics/{experimentName}/validation_stats_for_{property}.csv")

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
                save_to_path = f"checkpoints/{experimentName}/best_checkpoint_for_{property}.pt"
                
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

                saveCSV(pd.DataFrame(data=trainingStats), f"statistics/{experimentName}/training_stats_for_{property}.csv")
                saveCSV(pd.DataFrame(validationPredictions), f"statistics/{experimentName}/validation_stats_for_{property}.csv")

            else:
                bestLoss = bestLoss
            
            print(f"Validation mae error = {validPerformance}")
            
        print(f"validation took {validationTime}")
        
    trainEndingTime = time.time()
    totalTrainingTime = trainEndingTime-trainingStartingTime

    print("\n========== Training complete ========")
    print(f"Training orgoMol on {property} prediction took {timeFormat(totalTrainingTime)}")

    if taskName == "classification":
        print(f"The lowest roc score achieved on validation set on {property} is {bestRoc} at {bestEpoch}th epoch \n")

    elif taskName == "regression":
        print(f"The lowest mae error achieved on validation set on predicting {property} is {bestLoss} at {bestEpoch}th epoch \n")
    
    return trainingStats, validationPredictions

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

    saveCSV(pd.DataFrame(testPredictions), f"statistics/{experimentName}/test_stats_for_{property}.csv")
        
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
    # check if the GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Number of available devices: {torch.cuda.device_count()}')
        print(f'Current device is: {torch.cuda.current_device()}')
        print("Training and testing on", torch.cuda.device_count(), "GPUs!")
        print('-'*50)
    else:
        print("No GPU available, please connect to the GPU first or continue to use CPU instead")
        print('-'*50)
        device = torch.device("cpu")
        
    args = args_parser()
    config = vars(args)

    # set parameters
    batchSize = config.get('bs')
    maxLength = config.get('max_len')
    learningRate = config.get('max_len')
    dropRate = config.get('dr')
    epochs = config.get('epochs')
    warmupSteps = config.get('warmup_steps')
    preprocessingStrategy = config.get('preprocessing_strategy')
    tokenizerName = config.get('tokenizer')
    pooling = config.get('pooling')
    schedulerType = config.get('scheduler')
    normalizerType = config.get('normalizer')
    property = config.get('property_name')
    optimizerType = config.get('optimizer')
    taskName = config.get('task_name')
    trainDataPath = config.get('train_data_path')
    validDataPath = config.get('valid_data_path')
    testDataPath = config.get('test_data_path')
    experimentName = config.get('experimentName')
    
    # prepare the data
    trainData = preProcess(pd.read_csv(trainDataPath),preProcessingStrategy)
    validData = preProcess(pd.read_csv(validDataPath),preProcessingStrategy)
    testData = preProcess(pd.read_csv(testDataPath),preProcessingStrategy)
    
    # check property type to determine the task name (whether it is regression or classification)
    if trainData[property].dtype == 'bool':
        taskName = 'classification'

        #converting True->1.0 and False->0.0
        trainData[property] = trainData[property].astype(float)
        validData[property] = validData[property].astype(float) 
        testData[property] = testData[property].astype(float)  
    else:
        taskName = 'regression'
    
    trainLabelsArray = np.array(trainData[property])
    trainLabelsMean = torch.mean(torch.tensor(trainLabelsArray))
    trainLabelsStd = torch.std(torch.tensor(trainLabelsArray))
    trainLabelsMin = torch.min(torch.tensor(trainLabelsArray))
    trainLabelsMax = torch.max(torch.tensor(trainLabelsArray))
    
    # define loss functions
    maeLossFunction = nn.L1Loss()
    bceLossFunction = nn.BCEWithLogitsLoss()

    freeze = False # a boolean variable to determine if we freeze the pre-trained T5 weights

    # define the tokenizer
    if tokenizerName == 't5_tokenizer': 
        tokenizer = AutoTokenizer.from_pretrained("t5-small")

    elif tokenizerName == 'modified':
        tokenizer = AutoTokenizer.from_pretrained("tokenizers/t5_tokenizer_trained_on_modified_part_of_C4_and_textedge")

    # add defined special tokens to the tokenizer
    if pooling == 'cls':
        tokenizer.add_tokens(["[CLS]"])
    if preProcessingStrategy == 'Both' or 'Token':
        tokenizer.add_tokens('[NUM]')
        tokenizer.add_tokens('[ANG]')
        
    print('-'*50)
    print(f"train data = {len(trainData)} samples")
    print(f"valid data = {len(validData)} samples")
    print('-'*50)
    print(f"training on {getSequenceLenStats(trainData, tokenizer, maxLength)}% samples with whole sequence")
    print(f"validating on {getSequenceLenStats(validData, tokenizer, maxLength)}% samples with whole sequence")
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
    
    model = T5Full(baseModel, baseModelOutputSize, drop_rate= dropRate, pooling=pooling)
    

    # print the model parameters
    modelTrainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters = {modelTrainableParams}")
    
    # create dataloaders
    trainDataLoader = createDataLoaders(
        tokenizer, 
        trainData, 
        maxLength, 
        batchSize, 
        propertyValue=property, 
        pooling=pooling, 
        normalize=True, 
        normalizer=normalizerType
    )

    validDataLoader = createDataLoaders(
        tokenizer, 
        validData, 
        maxLength, 
        batchSize, 
        propertyValue=property, 
        pooling=pooling
    )

    testDataLoader = createDataLoaders(
        tokenizer, 
        testData, 
        maxLength, 
        batchSize, 
        propertyValue=property, 
        pooling=pooling
    )
    
    # define the optimizer
    if optimizerType == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr = learningRate
        )

    # set up the scheduler
    totalTrainingSteps = len(trainDataLoader) * epochs 
    if schedulerType == 'linear':
        scheduler = get_linear_schedule_with_warmup( #get_linear_schedule_with_warmup
            optimizer,
            numWarmupSteps= warmupSteps, #steps_ratio*total_training_steps,
            numTrainingSteps= totalTrainingSteps 
        )
    
    # from <https://github.com/usnistgov/alignn/blob/main/alignn/train.py>
    elif schedulerType == 'onecycle': 
        stepsPerEpoch = len(trainDataLoader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learningRate,
            epochs=epochs,
            steps_per_epoch=stepsPerEpoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    
    elif schedulerType == 'step':
         # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            stepSize=warmupSteps
        )
    
    elif schedulerType == 'lambda':
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
        
    model,optimizer = fabric.setup(model,optimizer)   
    trainDataLoader = fabric.setup_dataloaders(trainDataLoader)
    validDataLoader = fabric.setup_dataloaders(validDataLoader)
    testDataLoader = fabric.setup_dataloaders(testDataLoader)
    
    print("======= Training ... ========")
    training_stats, validation_predictions = train(model, optimizer, scheduler, maeLossFunction, maeLossFunction, 
        epochs, trainDataLoader, validDataLoader,normalizer=normalizerType)

    print("======= Evaluating on test set ========")
    bestModelPath = f"checkpoints/{experimentName}/best_checkpoint_for_{property}.pt" 
    
    bestModel = T5Full(baseModel, baseModelOutputSize, drop_rate =dropRate, pooling=pooling)
    bestModel = fabric.setup(bestModel)
    
    _, test_performance = evaluate(bestModel, maeLossFunction, testDataLoader, trainLabelsMean, trainLabelsStd, trainLabelsMin, trainLabelsMax, property, device, taskName, normalizer=normalizerType)