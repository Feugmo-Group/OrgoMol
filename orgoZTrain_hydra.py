import re
import glob
import time
import datetime
from datetime import timedelta
import tarfile
import os
import logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import SGD

# add the progress bar
from tqdm import tqdm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from tokenizers.pre_tokenizers import Whitespace

# Hydra imports
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# pre-defined functions
from orgoZModel import T5Full
from orgoZDataLoader import *
from ztok import ZTokenizer
from preProcess import *

# for metrics
from torchmetrics.classification import BinaryAUROC

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    output_path = where_to_save
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))  # Create the directory if it doesn't exist
    df.to_csv(output_path, index=False)
    
def getSequenceLenStats(df, tokenizer, max_len):
    training_on = sum(1 for sent in df['zmat_file'].apply(lambda x: tokenizer.tokenize(x)[0]) if len(sent) <= max_len)
    return (training_on/len(df))*100

def preprocess_data(df):
    # Drop rows with NaN values
    df = df.dropna()
    return df

def train(model, optimizer, scheduler, bceLossFunction, maeLossFunction, cfg, trainDataLoader, validDataLoader, device, 
          trainLabelsMean=None, trainLabelsStd=None, trainLabelsMin=None, trainLabelsMax=None):
    
    # Extract config parameters
    epochs = cfg.training.epochs
    property_name = cfg.property.name
    normalizer_type = cfg.normalizer.name
    task_name = cfg.experiment.task_name
    experiment_name = cfg.experiment.name
    max_grad_norm = cfg.training.max_grad_norm
    statistics_dir = cfg.paths.statistics_dir
    checkpoint_dir = cfg.paths.checkpoint_dir
    
    # Initialize tracking variables
    trainingStartingTime = time.time()
    trainingStats = []
    validationPredictions = {}
    
    bestLoss = 1e10
    bestRoc = 0.0
    
    for epoch in range(epochs):
        logger.info(f"========== Epoch {epoch + 1}/{epochs} =========")
        logger.info(f"Learning rate at epoch {epoch + 1}: {scheduler.get_last_lr()[0]}")  # Check learning rate

        epochStartingTime = time.time() 
        totalTrainingLoss = 0
        
        model.train()
        
        for step, batch in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader)):
            batchInputs, batchLabels, batchNormLabels, batchMasks, numericalInputs = tuple(b.to(device) for b in batch)

            _, predictions = model(batchInputs, batchMasks, numericalInputs)

            if task_name == 'classification':
                loss = bceLossFunction(predictions.squeeze(), batchLabels.squeeze())
                logger.debug(f"Step {step+1}, Loss: {loss.item()}")
            
            elif task_name == 'regression':
                loss = maeLossFunction(predictions.squeeze(), batchNormLabels.squeeze())
                
                if normalizer_type == 'z_norm':
                    predictionsDenorm = z_denormalize(predictions, trainLabelsMean, trainLabelsStd)
                elif normalizer_type == 'mm_norm':
                    predictionsDenorm = mm_denormalize(predictions, trainLabelsMin, trainLabelsMax)
                elif normalizer_type == 'ls_norm':
                    predictionsDenorm = ls_denormalize(predictions)
                elif normalizer_type == 'no_norm':
                    loss = maeLossFunction(predictions.squeeze(), batchLabels.squeeze())
                    predictionsDenorm = predictions

                maeLoss = maeLossFunction(predictionsDenorm.squeeze(), batchLabels.squeeze()) 
                logger.debug(f"Step {step+1}, MAE Loss: {maeLoss.item()}")
            
            if task_name == "classification":
                totalTrainingLoss += loss.item()
            elif task_name == "regression":
                totalTrainingLoss += maeLoss.item() if 'maeLoss' in locals() else loss.item()
            
            # back propagate
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # average training loss on actual output
        averageTrainingLoss = totalTrainingLoss/len(trainDataLoader) 
        
        epochEndingTime = time.time()
        trainingTime = timeFormat(epochEndingTime - epochStartingTime)

        logger.info(f'Average training loss = {averageTrainingLoss}')
        logger.info(f'Training for this epoch took {trainingTime}')

        logger.info("Running Validation ...")

        validStartTime = time.time()

        model.eval()
        
        predictionsList = []
        targetsList = []
        
        for step, batch in tqdm(enumerate(validDataLoader), total=len(validDataLoader)):
            batchInputs, batchLabels, batchNormLabels, batchMasks, numericalInputs = tuple(b.to(device) for b in batch)

            with torch.no_grad():
                _, predictions = model(batchInputs, batchMasks, numericalInputs)

                if task_name == "classification":
                    predictionsDenorm = predictions
                elif task_name == "regression":
                    if normalizer_type == 'z_norm':
                        predictionsDenorm = z_denormalize(predictions, trainLabelsMean, trainLabelsStd)
                    elif normalizer_type == 'mm_norm':
                        predictionsDenorm = mm_denormalize(predictions, trainLabelsMin, trainLabelsMax)
                    elif normalizer_type == 'ls_norm':
                        predictionsDenorm = ls_denormalize(predictions)
                    elif normalizer_type == 'no_norm':
                        predictionsDenorm = predictions

            predictions = predictionsDenorm.detach().cpu().numpy()
            targets = batchLabels.detach().cpu().numpy()
            
            for i in range(len(predictions)):
                predictionsList.append(predictions[i][0])
                targetsList.append(targets[i])
            
        validEndingTime = time.time()
        validationTime = timeFormat(validEndingTime-validStartTime)
            
        if task_name == "classification":
            validPerformance = getRocScore(predictionsList, targetsList)
            
            if validPerformance >= bestRoc:
                bestRoc = validPerformance
                bestEpoch = epoch+1

                # save the best model checkpoint
                save_to_path = f"{checkpoint_dir}/best_checkpoint_for_{property_name}.pt"
                if not os.path.exists(os.path.dirname(save_to_path)):
                    os.makedirs(os.path.dirname(save_to_path))  # Create the directory if it doesn't exist
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

                saveCSV(pd.DataFrame(data=trainingStats), f"{statistics_dir}/training_stats_for_{property_name}.csv")
                saveCSV(pd.DataFrame(validationPredictions), f"{statistics_dir}/validation_stats_for_{property_name}.csv")

            logger.info(f"Validation roc score = {validPerformance}")
            
        elif task_name == "regression":
            predictionsTensor = torch.tensor(predictionsList)
            targetsTensor = torch.tensor(targetsList)
            validPerformance = maeLossFunction(predictionsTensor.squeeze(), targetsTensor.squeeze())
        
            if validPerformance <= bestLoss:
                bestLoss = validPerformance
                bestEpoch = epoch+1
                
                # save the best model checkpoint
                save_to_path = f"{checkpoint_dir}/best_checkpoint_for_{property_name}.pt"
                if not os.path.exists(os.path.dirname(save_to_path)):
                    os.makedirs(os.path.dirname(save_to_path))  # Create the directory if it doesn't exist
                
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

                saveCSV(pd.DataFrame(data=trainingStats), f"{statistics_dir}/training_stats_for_{property_name}.csv")
                saveCSV(pd.DataFrame(validationPredictions), f"{statistics_dir}/validation_stats_for_{property_name}.csv")

            logger.info(f"Validation mae error = {validPerformance}")
            
        logger.info(f"validation took {validationTime}")
        
    trainEndingTime = time.time()
    totalTrainingTime = trainEndingTime-trainingStartingTime

    logger.info("\n========== Training complete ========")
    logger.info(f"Training orgoMol on {property_name} prediction took {timeFormat(totalTrainingTime)}")

    if task_name == "classification":
        logger.info(f"The lowest roc score achieved on validation set on {property_name} is {bestRoc} at {bestEpoch}th epoch \n")
    elif task_name == "regression":
        logger.info(f"The lowest mae error achieved on validation set on predicting {property_name} is {bestLoss} at {bestEpoch}th epoch \n")
    
    return trainingStats, validationPredictions

def evaluate(model, maeLossFunction, testDataLoader, trainLabelsMean, trainLabelsStd, trainLabelsMin, trainLabelsMax, cfg, device):
    property_name = cfg.property.name
    normalizer_type = cfg.normalizer.name
    task_name = cfg.experiment.task_name
    statistics_dir = cfg.paths.statistics_dir
    
    testStartTime = time.time()

    model.eval()

    totalTestLoss = 0
    predictionsList = []
    targetsList = []
    
    for step, batch in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        batchInputs, batchLabels, batchNormLabels, batchMasks, numericalInputs = tuple(b.to(device) for b in batch)

        with torch.no_grad():
            _, predictions = model(batchInputs, batchMasks, numericalInputs)

            if task_name == "classification":
                predictionsDenorm = predictions
            elif task_name == "regression":
                if normalizer_type == 'z_norm':
                    predictionsDenorm = z_denormalize(predictions, trainLabelsMean, trainLabelsStd)
                elif normalizer_type == 'mm_norm':
                    predictionsDenorm = mm_denormalize(predictions, trainLabelsMin, trainLabelsMax)
                elif normalizer_type == 'ls_norm':
                    predictionsDenorm = ls_denormalize(predictions)
                elif normalizer_type == 'no_norm':
                    predictionsDenorm = predictions

        predictions = predictionsDenorm.detach().cpu().numpy()
        targets = batchLabels.detach().cpu().numpy()

        for i in range(len(predictions)):
            predictionsList.append(predictions[i][0])
            targetsList.append(targets[i])
        
    testPredictions = {f"{property_name}": predictionsList}

    saveCSV(pd.DataFrame(testPredictions), f"{statistics_dir}/test_stats_for_{property_name}.csv")
        
    if task_name == "classification":
        testPerformance = getRocScore(predictionsList, targetsList)
        logger.info(f"\n The roc score achieved on test set for predicting {property_name} is {testPerformance}")
    elif task_name == "regression":
        predictionsTensor = torch.tensor(predictionsList)
        targetsTensor = torch.tensor(targetsList)
        testPerformance = maeLossFunction(predictionsTensor.squeeze(), targetsTensor.squeeze())
        logger.info(f"\n The mae error achieved on test set for predicting {property_name} is {testPerformance}")

    testEndingTime = time.time()
    testingTime = timeFormat(testEndingTime-testStartTime)
    logger.info(f"testing took {testingTime} \n")

    return predictionsList, testPerformance

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Print the config
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Convert relative paths to absolute paths
    train_data_path = to_absolute_path(cfg.paths.train_data_path)
    valid_data_path = to_absolute_path(cfg.paths.valid_data_path)
    test_data_path = to_absolute_path(cfg.paths.test_data_path)
    
    # Set random seed for reproducibility
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    
    # Check for property name
    if cfg.property.name is None:
        raise ValueError("Property name must be specified! Use property.name=<property_name>")
    
    # Determine device
    if cfg.experiment.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.experiment.device)
    
    if device.type == "cuda":
        logger.info(f'Number of available devices: {torch.cuda.device_count()}')
        logger.info(f'Current device is: {torch.cuda.current_device()}')
        logger.info(f"Training and testing on {torch.cuda.device_count()} GPUs!")
    else:
        logger.info("Training and testing on CPU")
    
    # Extract config parameters
    property_name = cfg.property.name
    batch_size = cfg.training.batch_size
    max_length = cfg.training.max_len
    learning_rate = cfg.optimizer.lr
    drop_rate = cfg.training.drop_rate
    epochs = cfg.training.epochs
    warmup_steps = cfg.scheduler.warmup_steps
    preprocessing_strategy = cfg.preprocessing.strategy
    tokenizer_name = cfg.tokenizer.name
    pooling = cfg.training.pooling
    scheduler_type = cfg.scheduler.name
    normalizer_type = cfg.normalizer.name
    freeze = cfg.training.freeze_encoder
    pct = cfg.scheduler.pct
    # Load and prepare data
    trainData = pd.read_csv(train_data_path)
    validData = pd.read_csv(valid_data_path)
    testData = pd.read_csv(test_data_path)
    
    # Preprocess
    trainData = preProcess(trainData, preprocessing_strategy)
    validData = preProcess(validData, preprocessing_strategy)
    testData = preProcess(testData, preprocessing_strategy)
    
    # Determine task type (classification or regression)
    if cfg.experiment.task_name == "auto":
        if trainData[property_name].dtype == 'bool':
            task_name = 'classification'
            # Converting True->1.0 and False->0.0
            trainData[property_name] = trainData[property_name].astype(float)
            validData[property_name] = validData[property_name].astype(float) 
            testData[property_name] = testData[property_name].astype(float)
        else:
            task_name = 'regression'
        
        # Update the config with detected task type
        cfg.experiment.task_name = task_name
    else:
        task_name = cfg.experiment.task_name
    
    trainLabelsArray = np.array(trainData[property_name])
    trainLabelsMean = torch.mean(torch.tensor(trainLabelsArray))
    trainLabelsStd = torch.std(torch.tensor(trainLabelsArray))
    trainLabelsMin = torch.min(torch.tensor(trainLabelsArray))
    trainLabelsMax = torch.max(torch.tensor(trainLabelsArray))
    
    # Define loss functions
    maeLossFunction = nn.L1Loss()
    bceLossFunction = nn.BCEWithLogitsLoss()
    
    # Initialize tokenizer
    if tokenizer_name == 'ztok':
        tokenizer = ZTokenizer()
    
    # Add defined special tokens to the tokenizer
    if pooling == 'cls':
        tokenizer.add_special_tokens(["[CLS]"])
    
    # Log data stats
    logger.info('-'*50)
    logger.info(f"train data = {len(trainData)} samples")
    logger.info(f"valid data = {len(validData)} samples")
    logger.info('-'*50)
    logger.info(f"training on {getSequenceLenStats(trainData, tokenizer, max_length)}% samples with whole sequence")
    logger.info(f"validating on {getSequenceLenStats(validData, tokenizer, max_length)}% samples with whole sequence")
    logger.info('-'*50)
    
    logger.info("labels statistics on training set:")
    logger.info(f"Mean: {trainLabelsMean}")
    logger.info(f"Standard deviation: {trainLabelsStd}")
    logger.info(f"Max: {trainLabelsMax}")
    logger.info(f"Min: {trainLabelsMin}")
    logger.info("-"*50)
    
    # Initialize model
    base_model = T5EncoderModel.from_pretrained(cfg.model.base_model)
    base_model_output_size = cfg.model.base_model_output_size
    
    # Freeze the pre-trained LM's parameters if specified
    if freeze:
        for param in base_model.parameters():
            param.requires_grad = False
    
    # Resize token embeddings
    base_model.resize_token_embeddings(len(tokenizer.vocab), pad_to_multiple_of=8)
    
    # Initialize the full model
    model = T5Full(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling)
    
    # Handle multi-GPU training if available
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        device_ids = [d for d in range(torch.cuda.device_count())]
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model.to(device)
    
    # Print model parameters
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters = {model_trainable_params}")
    
    # Create dataloaders
    trainDataLoader = createDataLoaders(
        tokenizer, 
        trainData,   
        batch_size, 
        property_name, 
        normalize=True, 
        normalizer=normalizer_type
    )

    validDataLoader = createDataLoaders(
        tokenizer, 
        validData,  
        batch_size, 
        property_name,
        normalize=False
    )

    testDataLoader = createDataLoaders(
        tokenizer, 
        testData,  
        batch_size, 
        property_name,
        normalize=False 
    )
    
    # Define optimizer
    if cfg.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
    
    # Set up scheduler
    total_training_steps = len(trainDataLoader) * epochs 
    
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps 
        )
    elif scheduler_type == 'onecycle': 
        steps_per_epoch = len(trainDataLoader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct,
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=warmup_steps
        )
    elif scheduler_type == 'lambda':
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    
    # Training phase
    logger.info("======= Training ... ========")
    training_stats, validation_predictions = train(
        model, optimizer, scheduler, bceLossFunction, maeLossFunction, 
        cfg, trainDataLoader, validDataLoader, device, 
        trainLabelsMean, trainLabelsStd, trainLabelsMin, trainLabelsMax
    )

    # Testing phase
    logger.info("======= Evaluating on test set ========")
    logger.info(f"Testing on {len(testData)} samples")
    
    checkpoint_path = f"{cfg.paths.checkpoint_dir}/best_checkpoint_for_{property_name}.pt"
    
    # Load best model for evaluation
    best_model = T5Full(base_model, base_model_output_size, drop_rate=drop_rate, pooling=pooling)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        best_model = nn.DataParallel(best_model, device_ids=device_ids).cuda()
        best_model.module.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    else:
        best_model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        best_model.to(device)
    
    # Evaluate on test set
    test_predictions, test_performance = evaluate(
        best_model, maeLossFunction, testDataLoader, 
        trainLabelsMean, trainLabelsStd, trainLabelsMin, trainLabelsMax, 
        cfg, device
    )
    
    return {
        "task_type": task_name,
        "property": property_name,
        "test_performance": float(test_performance),
        "best_model_path": checkpoint_path
    }

if __name__ == "__main__":
    main()