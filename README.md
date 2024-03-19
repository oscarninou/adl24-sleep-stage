# adl24-sleep-stage
Project for the course "Advanced Deep-Learning", in the 3rd year of ESPCI, 2024.

- This [notebook](./sleep-edf-data.ipynb) decribes the dataset, provides links and explanations
- The data are prepared with this [script](prepare_physionet.py)
- The prepared data can be downloaded from [this drive](https://drive.google.com/drive/folders/176qhDmYUDzQg5yrv-IPzkpKSCR3Jv083?usp=sharing).
- Look at [this notebook](./sleep-edf-npz.ipynb) to play with the raw data and create data esay to use.
- A pickle is available on the drive
  
     
To better start with, the data are pre-processed. In the directory `5-cassette` you will find the data subsampled at 20 kHZ, segmented in segments of 30s each. It could be easier to start with only the cassette part and then see how telemetry can be included in the study. 

The possible tasks can be split in different subprojects: what kind of models, of pre-processig, and evaluation of the performances.  
These subprojects are not mutually exclusive and they can be mixed. 


## Data representation 
Of course we can build model on the raw signal, but we can also think about more advanced representations. The first one could be the spectrum computed on a sliding window.  


## CNN models


The basic idea is to use CNN models. Some papers already try that. It can provide a good baseline. The CNN models can combine `Conv1D`, `MaxPooling`, and `BatchNorm`.  

## LSTM / GRU models

The next is to have a comparison between the following models:
- LSTM or GRU on the raw signal
- Bi-LSTM or bi-GRU
- Stacking bi-LSTM or bi-GRU
- Do the same but on the spectrums
- Do the same but on the results of convolutional layers (to reduce the length)

## Transformers 
To apply the transfomers we have similar variants for the data: 
- Starts from the raw signal,
- from the spectrum,
- from convolution: this is called [ConFormer](https://arxiv.org/abs/2005.08100) (proposed recently for speech processing).  

For each of them, there are many hyperparameters: the number of layers, the hidden dimensions, and the number of heads.


## Subject independancies
Starting from an already efficient architectures, is the model able to transfer its performance to an unseen subject. This can be tested by doing some leave-one-out experiments: 
- Take one subject, and remove it from the training data
- Train the model without this subject. 
- Evaluate and compare the performance. 

We can then see how many training segments of this subject is needed to close the gap of performance. 

## EOG vs EEG

The idea is to evaluate if the EOG files are sufficient for the prediction of the sleep stage. Do we need EEGs ? Is it better with both ? 


## Improve subject independancies with Adversarial training

[In this paper](https://jmlr.org/papers/volume17/15-239/15-239.pdf) (and especially in figure 1), they propose an adversarial training to make data representation more independant from some characteristics (the domain for text classification in the paper). We can do the same for the subject. Train jointly: 
- the classifier to label the sleep stage,
- the classifier to recognize the subject.


The encoder part, that is shared by these both classifiers, must 
be efficient for the sleep stage, but the second should not be able to recognize the subject .




## Pre-training 
Assume we have all the cassette signals, but the labels are available only for a reduced subparts, let say $N\%$. One idea to leverage the unanotated part is to find a way to pre-train the encoder part of the network: 

- A first solution is to train an auto-encoder (see the project of 2A). 
- A first kind of masked LM: reconstruct a masked part of the signal
- A second kind of masked LM: discretized first what we want to reconstruct, to recast the reconstruction as a classification 

For the last one, this what is proposed to process speech signals by the [last state-of-the-art model Best-RQ](https://arxiv.org/pdf/2202.01855.pdf). In this paper they use transformers. 




