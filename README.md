# OrgoMol
LLM that uses text-decription of organic molecules to predict properties and generate molecules

Currently most of the raw_data and csv file containing the training data sit in a onedrive. They are too big to be put on GitHub
one can use these scripts to generate that data and maybe in the future I will detail how that can be done

It is important once you have converted your xyz files to zmat and then to txt files to "validate" your data. This is what the sanityCheck script does.
At the moment it runs sequentially, a huge draw back as it takes about 30 minutes to run on 130,000 files. In the future it would be beneficial to parallelize this.

Validation, in this sense, is essentially converting your zmat files back to xyz files. Computing a distance matrix for both files. Then comparing their norms and ensuring they agree to some tolerance


this work here is based on the LLM-prop model trained by Vertaix at Princeton.

@article{rubungo2023llm,
  title={LLM-Prop: Predicting Physical And Electronic Properties Of Crystalline Solids From Their Text Descriptions},
  author={Rubungo, Andre Niyongabo and Arnold, Craig and Rand, Barry P and Dieng, Adji Bousso},
  journal={arXiv preprint arXiv:2310.14029},
  year={2023}
}

## Basic Workflow Outline

1. Aquire xyz files from any dataset. Ensure that the xyz files have first line as the number of atoms and then the coordinates with nothing inbetween
2. Run the namesToList script in the dataset directory to generate a list of all the files in the dataset to feed into ptqdm (not currently working as intended)
3. Run the zToText script in the dataset directory to convert xyz files to zmat and text files, using the list of names from previous step
4. Run txtToCsv to store all the text files in a csv files
5. Use getProperties to use xyz files with properties from qm9 or whatever database to associate txt files with given properties
6. Conduct sanityCheck at some point to validate data
7. split dataset into training, validaton and test sets using sampler script
8. Running orgoMolTraining script with correct paths to training, validation and test sets
9. 

## Current HyperParameters

- batchSize = 8
- maxLength = 512
- learningRate = 1E-4
- dropRate = 0.5
- epochs = 200
- warmupSteps = 10
- preprocessingStrategy = config.get('preprocessing_strategy')
- tokenizerName = 't5_tokenizer'
- pooling = 'cls'
- schedulerType = 'onecycle'
- normalizerType = 'z_norm'
- property = "homoLumoGap"
- optimizerType = "adamw"
- taskName = "Regression"


## Currently Working On:

 - ~~Training and tuning hyperparamaters~~
 - ~~Changing property units to eV~~
 - ~~Train on bigger dataset~~
-  Changing how zToText works to create a more natural language problem
 - ~~Using preprocessing techniques on data~~

### Future 

 - Fixing namesToList
 - Fixing xyz conversion so not so specific
 - making getProperties get more properties
 - adding support for smile conversion
- Using rdkit to get functional groups




