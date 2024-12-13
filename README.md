# XML models
## CascadeXML
CascadeXML requires five input files to run:
- **train_raw_texts.txt** - This file contains all the texts used for training. Texts are seperated by a newline character.
- **test_raw_texts.txt** - This file contains all the texts used for testing. Texts are seperated by a newline character.
- **Y.trn.txt** - This file contains all the labels used for training. Each line contains a set of numbers. These are the numbers of the labels, which are relevant for the text, situated in the same line in the train_raw_texts file.
- **Y.tst.txt** - This file contains all the labels used for testing. Each line contains a set of numbers. These are the numbers of the labels, which are relevant for the text, situated in the same line in the test_raw_texts file.
- **train.txt** - An additional input file used for clustering. For each data point, it contains the numbers of all relevant labels, followed by TFIDF feature representations of the input text.

For datasets from the world of XML, the first four files can be downloaded from [here](https://github.com/yourh/AttentionXML). The train.txt file can be downloaded from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), by selecting the "BOW Features" mirror for the desired dataset. The source code and installation instructions can be obtained from the [original git repository](https://github.com/xmc-aalto/cascadexml).

The NYT, RCV1-V2 and [WoS](https://data.mendeley.com/datasets/9rw3vkcfy4/6) datasets, must be downloaded and parsed into the format described above to obtain the first four files. The train.txt files for each dataset can be dowloaded from [here](https://drive.google.com/drive/folders/1dHqrKTVkjPvZ0ozlOu9UUOJykW64tVXW?usp=sharing). Note that the dimensions of the dataset must fit the dimension of the train.txt file. We used the following versions: NYT: 29179 training datapoints, RCV1: 23149 training datapoints, WOS: 37588 training datapoints.

Alternatively, for custom datasets, these train.txt files can be created manually by untilising the scripts in the CascadeXML/preprocessing folder. To do this, donwload the folder, place the train_raw_texts.txt file and Y.trn.txt files for your dataset in the same folder and then run the xml_prepocessor.py file. This should create 2 more files named text_preprocessed.txt and vocab.txt. When these are created, run the vectorizer.py file. This should then create the train.txt file for your dataset. Note that the vectorizer.py script may take several hours to terminate. 

## XR-Transformer
XR-Transformer requires six input files:
- **X.trn.txt** - This file contains the texts used for training. Texts are seperated by a newline character.
- **X.trn.npz** - This file is a CSR npz or Row-majored npy file and contains the training feature matrix with shape N x d (N: number of training intances, d: number of feauture dimensions).
- **Y.trn.npz** - This file is a CSR npz file with the training label matrix containing the ground truth label assignment for the training. Shape: N x L (L: number of labels in label space).
- **X.tst.txt** - This file contains the texts used for testing/prediction. Texts are seperated by a newline character.
- **X.tst.npz** - This file is a CSR npz or Row-majored npy file containing the testing feature matrix with shape N x d.
- **Y.tst.npz** - This file is a CSR npz files of the testing label matrix containing the ground truth label assignment for the test data. Shape N x L.

The datasets from the XML world are provided by the authors of XR-Transformer and can be downloaded [here](https://ia902308.us.archive.org/21/items/pecos-dataset/xmc-base/).

#### Preprocessing of foreign datasets 

The HTC datasets (NYT, RCV1-V2 and WoS) can be preprocessed using the provided XR-Transformer [preprocessing.py](https://github.com/FloHauss/XMC_HTC/blob/main/XR-Transformer/preprocessing/preprocess.py) script in /XR-Transformer/preprocessing. The script needs the following input files: "train_raw_texts.txt" and "test_raw_texts.txt" containing the raw text train and test data, as well as "train_labels.txt" and "test_labels.txt" containing the train and test labels. Note, that these files are similar to the inputs of the CascadeXML model ("train_labels.txt" = "Y.trn.txt", "test_labels.txt" = "Y.tst.txt"). The outputs of the preprocessing script are the .npz files required for the execution of XR-Transformer. They need to be renamed and structured as described below.

In order to use the scripts we provided, the datasets need to be stored in XMLmodels/pecos like this:
```
|-- xmc-base
|   |-- tfidf-attnxml
|   |   |-- X.trn.npz
|   |   |-- X.tst.npz
|   |-- X.trn.txt
|   |-- X.tst.txt
|   |-- Y.trn.txt
|   |-- Y.tst.txt
````
#### Start experiments

First XR-Transformer needs to be installed ([see xr_transformer_guide](https://github.com/FloHauss/XMC_HTC/blob/main/xr_transformer_guide.md)). To conduct our experiments we used [run_ensemble/run.sh](https://github.com/FloHauss/XMC_HTC/blob/main/XMLmodels/pecos/run_ensemble/run.sh). It can be called like this:
```sh
bash run.sh ${DATASET} ${PATH_TO_DATASET}

# DATASET: name of the dataset (e.g. "wiki10-31k"). Used to identify data and parameter files.
# PATH_TO_DATASET: folder in which the dataset folder is stored (e.g. "xmc-base" or "htc-base")
```

#### Results
The logs and results of our experiments can be found in [run_ensemble/results/](https://github.com/FloHauss/XMC_HTC/tree/main/XMLmodels/pecos/run_ensemble/results).

#### Adjustments of the model
The changes we made in comparison to the XR-Tranformer model by amazon are mainly due to introducing the R-Precision metric, introduced in [precos/utils/smat_util.py](https://github.com/FloHauss/XMC_HTC/blob/main/XMLmodels/pecos/pecos/utils/smat_util.py). They are marked in the code with comments in the format "changed: ...". 

# HTC models

### XML datasets for HTC
As HGCLR does not support larger datasets without heavy modifications, only HBGL makes use of the larger XML datasets.
Of these, even with modifications within the HBGL source code to accomodate for larger datasets, only Wiki10-31k and Ammazoncat-13k are small enough to effectively run on them.

So while our frameworks supports conversion for these larger datasets the model wills not effectively run on them.

## HBGL 
HBGL requires 4 input files to run:
- {dataset_name}.taxonomy
- {dataset name}_test.json
- {dataset_name}_train.json
- {dataset_name}_val.json

It is required that these files are located within a folder named {dataset_name} within the data folder of HBGL.
Each dataset also requires a corresponding script file named {dataset_name}.sh within the scripts folder of HBGL. HBGL runs are also started from within this folder.

### HTC datasets for HBGL
We follow the same preprocessing procedure as described by [HBGL](https://github.com/kongds/HBGL).
Use the respective preprocessing files within the corresponding dataset folder.
We made some minor adjustments, as otherwise the preprocessing will not work:
- For WoS this includes an additional preprocessing file that has to run before the actual preprocess. This is because the seperator within the dataset file is not recognized properly
- For NYT no additional changes were required
- For RCV1 we adjusted the preprocessing file to properly work with the respective file paths

### XML datasets for HBGL
Within the data_transfer folder the specific XML dataset has to be inserted within the specific input/xml/{dataset_name} folder.
6 files are required:
- {dataset_name}.taxonomy
- {dataset_name}_label_map.txt
- {dataset_name}_test_labels.txt
- {dataset_name}_test_texts.txt
- {dataset_name}_train_labels.txt
- {dataset_name}_train_texts.txt

The test and train files are the same files used in Cascade XML just renamed.
The taxonomies can be found [here](https://drive.google.com/drive/folders/18qnFBX67E1lldpJNPXZa8_I0Tylpl_cU). The labels are clustered using k-means and these clusters then provide the hierachical structure.
The label map file contains all labels sorted alphabetically. This [repository](http://manikvarma.org/downloads/XC/XMLRepository.html) contains most the relevant XML datasets where this file is typically named something like 'Y.txt'.

## HGCLR
HGCLR requires 2 input files to run:\
tok.txt: This file contains tokenized text data. Each number in the file represents a token ID corresponding to words or subwords from the original dataset.\
Y.txt: This file contains all hierarchical labels. Each line contains the label vector for a single instance. 

It is necessary that the files for each dataset are located in their respective directories. For example, files for the WoS dataset should be stored in the wos folder, and files for the RCV1 dataset should be stored in the rcv1 folder. This ensures that the scripts can correctly access the required data.\
The optimal training parameters for each dataset can be found in the [here](https://github.com/wzh9969/contrastive-htc). 

# Guides
- use BwUniCluster: [guide here](bw_uni_cluster.md) (written in german, and only for members of applicable universaties)
- install and run XR-Transformer: [guide here](xr_transformer_guide.md)
