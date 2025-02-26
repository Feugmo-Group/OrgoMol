import os
import pandas as pd
import logging
from tqdm import tqdm

# Hydra imports
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Import preprocessing functions
from preProcess import preProcess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Preprocess data files using the configuration from Hydra.
    This script takes data files, applies the specified preprocessing strategy,
    and saves the processed data to output files.
    """
    # Print the config
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Get data paths
    train_data_path = to_absolute_path(cfg.paths.train_data_path)
    valid_data_path = to_absolute_path(cfg.paths.valid_data_path)
    test_data_path = to_absolute_path(cfg.paths.test_data_path)
    
    # Get preprocessing strategy
    strategy = cfg.preprocessing.strategy
    
    # Define output paths
    output_dir = cfg.paths.get('processed_data_dir', 'processed_data')
    output_dir = to_absolute_path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    train_output_path = os.path.join(output_dir, 'train_processed.csv')
    valid_output_path = os.path.join(output_dir, 'valid_processed.csv')
    test_output_path = os.path.join(output_dir, 'test_processed.csv')
    
    # Load data
    logger.info(f"Loading train data from {train_data_path}")
    train_data = pd.read_csv(train_data_path)
    
    logger.info(f"Loading validation data from {valid_data_path}")
    valid_data = pd.read_csv(valid_data_path)
    
    logger.info(f"Loading test data from {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    
    # Apply preprocessing
    logger.info(f"Applying preprocessing strategy: {strategy}")
    
    logger.info("Preprocessing training data...")
    train_processed = preProcess(train_data, strategy)
    
    logger.info("Preprocessing validation data...")
    valid_processed = preProcess(valid_data, strategy)
    
    logger.info("Preprocessing test data...")
    test_processed = preProcess(test_data, strategy)
    
    # Save processed data
    logger.info(f"Saving processed training data to {train_output_path}")
    train_processed.to_csv(train_output_path, index=False)
    
    logger.info(f"Saving processed validation data to {valid_output_path}")
    valid_processed.to_csv(valid_output_path, index=False)
    
    logger.info(f"Saving processed test data to {test_output_path}")
    test_processed.to_csv(test_output_path, index=False)
    
    logger.info("Preprocessing complete!")
    
    # Update config with new paths
    cfg.paths.train_data_path = train_output_path
    cfg.paths.valid_data_path = valid_output_path
    cfg.paths.test_data_path = test_output_path
    
    # Save updated config
    updated_config_path = os.path.join(output_dir, 'config.yaml')
    with open(updated_config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    logger.info(f"Updated config saved to {updated_config_path}")
    
    return {
        "train_path": train_output_path,
        "valid_path": valid_output_path,
        "test_path": test_output_path,
        "config_path": updated_config_path
    }

if __name__ == "__main__":
    main()