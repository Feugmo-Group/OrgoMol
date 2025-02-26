# OrgoMol with Hydra Configuration

This repository contains code for training and evaluating T5 based architecture for molecular property prediction, using Hydra for configuration management.

## Setup

1. Install the required dependencies:
```bash
pip install torch transformers hydra-core omegaconf tqdm pandas numpy torchmetrics
```

2. Ensure your data is in the correct format. The model expects CSV files with columns including:
   - `zmat_file`: Path to the Z-matrix file for each molecule
   - Your target property column (specified in the config)

## Directory Structure

```
.
├── config/                      # Hydra configuration files
│   ├── config.yaml              # Main configuration
│   ├── data/                    # Data configurations
│   ├── experiment/              # Experiment configurations
│   ├── model/                   # Model configurations
│   ├── normalizer/              # Normalizer configurations
│   ├── optimizer/               # Optimizer configurations
│   ├── scheduler/               # Scheduler configurations
│   └── tokenizer/               # Tokenizer configurations
├── orgoZModel.py                # Model architecture
├── orgoZDataLoader.py           # Data loading utilities
├── orgoZTrain_hydra.py          # Training script with Hydra
├── orgoZEvaluate_hydra.py       # Evaluation script with Hydra
├── preProcess.py                # Text preprocessing utilities
├── preProcess_hydra.py          # Preprocessing script with Hydra
└── ztok.py                      # Custom tokenization for molecular data
```

## Using Hydra for Configuration

### Basic Usage

To train a model with the default configuration:

```bash
python orgoZTrain_hydra.py property.name=your_property_name
```

This will create an output directory in `outputs/` with the experiment name and timestamp. All logs and results will be saved there.

### Overriding Configuration Values

You can override any configuration value from the command line:

```bash
python orgoZTrain_hydra.py property.name=your_property_name training.batch_size=32 optimizer.lr=5e-5
```

### Using Different Configuration Files

To use a different set of configurations:

```bash
python orgoZTrain_hydra.py experiment=custom_experiment model=t5_large
```

### Creating New Configuration Files

1. Create a new YAML file in the appropriate config subdirectory
2. Use it by specifying its name without the `.yaml` extension

Example for a new optimizer configuration (`config/optimizer/sgd.yaml`):
```yaml
# @package _group_
name: sgd
lr: 0.01
momentum: 0.9
```

Then use it with:
```bash
python orgoZTrain_hydra.py optimizer=sgd
```

### Multi-run Experiments

To run multiple experiments with different configurations:

```bash
python orgoZTrain_hydra.py --multirun optimizer.lr=1e-4,1e-5,1e-6
```

This will run the experiment three times with different learning rates.

## Preprocessing Data

To preprocess data before training:

```bash
python preProcess_hydra.py preprocessing.strategy=All
```

This will apply the specified preprocessing strategy to your data and save the processed files.

## Evaluation

To evaluate a trained model:

```bash
python orgoZEvaluate_hydra.py property.name=your_property_name
```

By default, it will look for the best checkpoint in the directory specified in your config.

To evaluate a specific checkpoint:

```bash
python orgoZEvaluate_hydra.py property.name=your_property_name checkpoint_path=/path/to/checkpoint.pt
```

## Configuration Structure

The configuration is structured hierarchically:

- **experiment**: General experiment settings (name, seed, etc.)
- **paths**: Data and output paths
- **property**: Target property to predict
- **training**: Training hyperparameters
- **validation**: Validation settings
- **testing**: Testing settings
- **model**: Model architecture settings
- **optimizer**: Optimizer settings
- **scheduler**: Learning rate scheduler settings
- **tokenizer**: Tokenizer settings
- **normalizer**: Data normalization settings
- **preprocessing**: Text preprocessing settings

See the configuration files for more details and options.
