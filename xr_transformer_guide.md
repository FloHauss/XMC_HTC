# XR-Transformer
## How to install and run XR-Transformer
*Author: Leon*

### 1. Clone this repository with submodule XR-Transformer
To do this either

- Clone this repository with submodules:
```sh
git clone --recurse-submodules https://github.com/FloHauss/XMC_HTC.git
```
**or** 
- Add submodules after cloning (run this in root directory of repository)
```sh
git pull
git submodule update --init
```

### 2. Install miniconda and setup virtual environment
- Follow instructions on anaconda docs [here](https://docs.anaconda.com/free/miniconda/index.html) in section "Quick command line install". 
- create a conda environment with python version 3.9 and activate it.
```sh
    conda create --name xr_transformer_env python=3.9
    conda activate xr_transformer_env
```

### 3. Install XR-Transformer in conda environment
- navigate to XMLmodels/pecos and install required packages
```sh
cd XMLmodels/pecos
python3 -m pip install --editable ./
```

### 4. Download XML datasets
Download datasets from XR-Transformer and save them into the folder xmc-base:
```sh
# wiki10-31k, amazoncat-13k, amazon-670k, wiki-500k, amazon-3m
DATASET="wiki10-31k"
wget https://archive.org/download/pecos-dataset/xmc-base/${DATASET}.tar.gz
tar -zxvf ./${DATASET}.tar.gz
```

Replace the DATASET-variable with the dataset you want to download. 

### 5. Train XR-Transformer (with specific Dataset)
Run command in directory 'pecos'.

Wiki10-31k:
```sh
python3 -m pecos.xmc.xtransformer.train -t ./xmc-base/wiki10-31k/X.trn.txt -x ./xmc-base/wiki10-31k/tfidf-attnxml/X.trn.npz -y ./xmc-base/wiki10-31k/Y.trn.npz -m ./trained-models/xr_model_wiki10-31k
```

### 6. Test model (with specific Dataset)
Run command in directory 'pecos'.

Wiki10-31k:
```sh
python3 -m pecos.xmc.xtransformer.predict -t ./xmc-base/wiki10-31k/X.tst.txt -x ./xmc-base/wiki10-31k/tfidf-attnxml/X.tst.npz -m ./trained-models/xr_model_wiki10-31k -o ./predictions/xr_prediction_wiki10-31k
```

### 7. Evaluate predictions (with specific Dataset)
Run command in directory 'pecos'.

Wiki10-31k:
```sh
python3 -m pecos.xmc.xlinear.evaluate -y ./xmc-base/wiki10-31k/Y.tst.npz -p ./predictions/xr_prediction_wiki10-31k -k 10
```

Returns precision@k and recall@k for each value up to k. You can specify k as parameter after '-k' (here it is set to 10).



### Use sbatch 
Here is what the a job script (e.g. job.sh) could look like:

```sh
#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --time=8:00:00
#SBATCH --mem=80000
#SBATCH --job-name=xrtransformer_amazoncat-13k
#SBATCH --gres=gpu:1

source /home/ul/ul_student/ul_ruw26/.bashrc 
# replace "/home/ul/ul_student/ul_ruw26/" with the path to .bashrc in your system
conda activate xr_transformer_env

python3 -m pecos.xmc.xtransformer.train -t ./xmc-base/amazoncat-13k/X.trn.txt -x ./xmc-base/amazoncat-13k/tfidf-attnxml/X.trn.npz -y ./xmc-base/amazoncat-13k/Y.trn.npz -m ./trained-models/xr_model_amazoncat-13k
```

**Important**: 
- without the "source"-command the shell doesn't know your conda environemt
- Don't use "partition=gpu_4_**h100**", since the cuda version used in xr-transformer is not compatible with the h100 gpu

submit it via:
```sh
sbatch job.sh
```

