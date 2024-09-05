# XML models
## CascadeXML
CascadeXML requires five input files to run: \
train_raw_texts.txt - This file contains all the texts used for training. Texts are seperated by a newline character.\
test_raw_texts.txt - This file contains all the texts used for testing. Texts are seperated by a newline character.\
Y.trn.txt - This file contains all the labels used for training. Each line contains a set of numbers. These are the numbers of the labels, which are relevant for the text, situated in the same line in the train_raw_texts file.\
Y.tst.txt - This file contains all the labels used for testing. Each line contains a set of numbers. These are the numbers of the labels, which are relevant for the text, situated in the same line in the test_raw_texts file.\
train.txt - An additional input file used for clustering. For each data point, it contains the numbers of all relevant labels, followed by TFIDF feature representations of the input text.

For datasets from the world of XML, the first four files can be downloaded from [here](https://github.com/yourh/AttentionXML). The train.txt file can be downloaded from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), by selecting the "BOW Features" mirror for the desired dataset. 

The NYT, RCV1-V2 and [WoS](https://data.mendeley.com/datasets/9rw3vkcfy4/6) datasets, must be downloaded and parsed into the format described above to obtain the first four files. The train.txt files for each dataset can be dowloaded from [here](https://drive.google.com/drive/folders/1dHqrKTVkjPvZ0ozlOu9UUOJykW64tVXW?usp=sharing). Note that the dimensions of the dataset must fit the dimension of the train.txt file. We used the following versions: NYT: 29179 training datapoints, RCV1: 23149 training datapoints, WOS: 37588 training datapoints.

Alternatively, for custom datasets, these train.txt files can be created manually by untilising the scripts in the PLACEHOLDER folder. To do this, donwload the folder, place the train_raw_texts.txt file and Y.trn.txt files for your dataset in the same folder and then run the xml_prepocessor.py file. This should create 2 more files named text_preprocessed.txt and vocab.txt. When these are created, run the vectorizer.py file. This should then create the train.txt file for your dataset. Note that the vectorizer.py script may take several hours to terminate. 
## XR-Transformer

# HTC models
## HBGL 
HBGL requires 3 input files to run:
- [dataset name]_test.json
- [dataset_name]_train.json
- [dataset_name]_val.json
It is required that these files are located within a folder named [dataset_name] within the data folder of HBGL.
Each dataset also requires a corresponding script file named [dataset_name].sh within the scripts folder of HBGL. Runs are also started from within this folder.
- TODO: NYT UND RCV1 PREPROCESSING WAREN PROBLEMATISCH???

### XML to HTC
As HGCLR does not support larger datasets without heavy modifications, only HBGL makes use of the larger XML datasets.
Of these, even with modifications within the HBGL source code to accomodate for larger datasets, only Wiki10-31k and Ammazoncat-13k are small enough to effectively run on them.
- TODO: Ordner und Files einbinden und nochmal beschreiben

## HGCLR

# Guides
- use BwUniCluster: [guide here](bw_uni_cluster.md)
- install and run XR-Transformer: [guide here](xr_transformer_guide.md)
