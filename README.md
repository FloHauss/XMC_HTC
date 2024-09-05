# XML models
## CascadeXML
CascadeXML requires five input files to run: \
train_raw_texts.txt - This file contains all the texts, used for training.\
test_raw_texts.txt - This file contains all the texts, used for testing.\
Y.trn.txt - This file contains all the labels, used for training.\
Y.tst.txt - This file contains all the labels, used for testing.\
train.txt - An additional input file, which is used for clustering. For each datapoint, it contains the numbers of all relevant labels, followed by TFIDF-feature representations of the input text.

For datasets from the world of XML, the first four files can be downloaded from [here](https://github.com/yourh/AttentionXML). The train.txt file can be downloaded from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), by selecting the "BOW Features" mirror for the desired dataset. 

For the NYT, RCV1-V2 and [WoS](https://data.mendeley.com/datasets/9rw3vkcfy4/6) datasets, 
## XR-Transformer

# HTC models
## HBGL

## HGCLR

# Guides
- use BwUniCluster: [guide here](bw_uni_cluster.md)
- install and run XR-Transformer: [guide here](xr_transformer_guide.md)
