# OrgoMol
LLM that uses text-decription of organic molecules to predict properties and generate molecules

Currently most of the raw_data and csv file containing the training data sit in a onedrive. They are too big to be put on GitHub
one can use these scripts to generate that data and maybe in the future I will detail how that can be done

It is important once you have converted your xyz files to zmat and then to txt files to "validate" your data. This is what the sanityCheck script does.
At the moment it runs sequentially, a huge draw back as it takes about 30 minutes to run on 130,000 files. In the future it would be beneficial to parallelize this.

Validation, in this sense, is essentially converting your zmat files back to xyz files. Computing a distance matrix for both files. Then comparing their norms and ensuring they agree to some tolerance


this work here is based on the LLM-prop model trained by Vertaix at Princeton.
Link to the paper: https://arxiv.org/pdf/2310.14029

## Currently Working On:
writing data loaders and training script


