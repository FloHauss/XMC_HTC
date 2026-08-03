# Config Options
---
Our used configs are provided and can be used. We will list some of the most important parameters and explain their behavior. Some of them are only available for certain model variants:

- name `string`:  Description of the run. not required.
- dataset `string`: Mandatory, abbreviation of the dataset that needs to match with the data folder
- model_variant `string`: 'radar' or 'roberta' for RADAr++ and the RoBERTa baseline, respectively
- loss `string`: 'focal' for RADAr++, 'bce' for RoBERTa
- decoding_strategy `string`: either 'greedy' or 'beam' for RADAr++, 'threshold' for RoBERTa

RADAr++
- beam_size `int`: only for 'beam', determines size of the beam
- length_penalty `float`: only for 'beam', determines the penalty put on longer sequences
RoBERTa
- threshold `float`: threshold value for 'threshold', determining what tokens are selected

- hyperparameter_tuning `bool`: For RADAr++ this enables Focal Temperature Scaling, for RoBERTa it tries to optimize the threshold.
- encoder `string`: the hugging-face model string for pre-trained models. 
- max_length `int`: the maximum number of input tokens accpeted by the encoder
- hidden_dim `int`: the dimension of encoder hidden layers
- dropout `float`: the value of the dropout

RADAr++
- embedding_size `int`: the dimension of decoder hidden layers
- forward_expansion `int`: factors with the embedding size. Forward expansion inside the decoder
- nhead `int`: number of attention heads within a decoder layer
- decoder_layers `int`: number of layers within the decoder

batch_size `int`: size of batches
epochs `int`: number of epochs
accumulation_steps: `int`: for gradient accumulation. batch size 32 with accumulation of 2 emulates batch size 64
lr_encoder `float`: learning rate of the encoder

RADAr++
- lr_decoder `float`: learning rate of the decoder

- lr_patience `int`: number of epochs without improvement before adjusting the learning rate factor
- patience `int`: number of epochs before early stopping

RADAr++
- gamma `float`: value of the focal calibration parameter gamma
- label_smoothing `float`: value for label smoothing within the focal loss calculation