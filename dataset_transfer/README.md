# How to use:

## XML to HTC
### input folder:
- within a specific dataset folder, e.g. 'input/xml/wiki10-31k', 6 files are required:
    - the taxonomy file: '{dataset_name}.taxonomy'
    - the label map: '{dataset_name}_label_map.txt'
    - the test labels: '{dataset_name}_test_labels.txt'
    - the test texts: '{dataset_name}_test_texts.txt'
    - the train labels: '{dataset_name}_train_labels.txt'
    - the train texts: '{dataset_name}_train_texts.txt'
- the taxonomy is pre-generated. [Number_Clusterings !](https://drive.google.com/drive/folders/1f9ZoAEzkVxxcBUdYbWfpSY1Cmf8O5ON_)
- the label maps can be found within the [XMLRepository](http://manikvarma.org/downloads/XC/XMLRepository.html)
    - they are within the BoW Features for the specific dataset and typically named 'Y.txt'
- the test and train text files can be found [here](https://github.com/yourh/AttentionXML) under 'Datasets' in the README
- it is required to rename these files to match the above mentioned file names

### transfer:
- use 'python3 xml_to_htc dataset_name' to convert the XML-format dataset to an HTC-format one

### output folder:
- within a specific dataset folder, e.g. 'output/xml/wiki10-31k', 4 files should have been generated:
    - the taxonomy file: '{dataset_name}.taxonomy' (Notice that this file is effectivley a copy of the original .taxonomy file)
    - the train file: '{dataset_name}_train.json'
    - the val file: '{dataset_name}_val.json'
    - the test file: '{dataset_name}_test.json'
- these files can be put into the specific data folders within HBGL

## HTC to XML
### input folder:
- wihtin a specific dataset folder, e.g. 'input/htc/wos', 4 files are required:
    - the taxonomy file: '{dataset_name}.taxonomy'
    - the train file: '{dataset_name}_train.json'
    - the val file: '{dataset_name}_val.json'
    - the test file: '{dataset_name}_test.json'
- these files can be obtained by following the preprocessing steps for HBGL (described within the initital README.md)

### transfer:
- use 'python3 htc_to_xml dataset_name' to convert the HTC-format dataset to an XML format one
- you can give an additional input parameter called 'leaves_only' leading to only the most specific labels of a datapoint being transferred. E.g. a datapoint with label A and label B might only contain label B after the process IF label A is a predecessor of B. 
    
    This trimming of the dataset might be interesting in some cases, as it reduces the dataset down to the essential labels. "Lost" predecessor labels can be restored by looking them up in the original hierarchy file.

- you can also call 'python3 htc_to_htc_lite dataset_name' to convert the HTC-format dataset to an HTC-format dataset were this modificatin is applied

### out folder:
- within a specific dataset folder, e.g. 'input/xml/wiki10-31k', 5 files should have been generated:
    - the id map: 'id_map.txt' (this files sorts each label within the dataset alphabetically and maps its position as its id. E.g. the first label of wos 'Addiction' will be mapped to 0)
    - the test labels: 'test_labels.txt'
    - the test texts: 'test_texts.txt'
    - the train labels: 'train_labels.txt'
    - the train texts: 'train_texts.txt'
- you can look into the initial README.md on how to include these converted files for the respective XML model