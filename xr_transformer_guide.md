# XR-Transformer
## How to install XR-Transformer
*Author: Leon*

### 1. Clone this repository
```sh
git clone https://github.com/FloHauss/XMC_HTC.git
```

### 2. Install miniconda and setup virtual environment
Follow instructions on anaconda docs [here](https://docs.anaconda.com/free/miniconda/index.html) in section "Quick command line install". \
Create a conda environment with python version 3.9 and activate it:
```sh
conda create --name xr_transformer_env python=3.9
conda activate xr_transformer_env
```

### 3. Install XR-Transformer in conda environment
Navigate to XMLmodels/pecos and install required packages:
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
