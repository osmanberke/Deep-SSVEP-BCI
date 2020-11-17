# A Deep Neural Network for SSVEP-based Brain Computer Interfaces
This is the official repository for deep neural networks described in paper: https://arxiv.org/abs/ 
This repository alows you to train and test models to replicate the results

|![alt text](https://github.com/osmanberke/Deep-SSVEP-BCI/blob/main/system.png)|
|:--:| 
|*Proposed deep learning model architecture. It is drawn for input signal that has 50 time samples from 9 channels and is filtered with 3 bandpass filters. When input size or utilized number of bandpass filters change, corresponding kernels' sizes change accordingly.*|


# Requisites

All models were implemented tested with MATLAB 2018B. All following steps assume that they are executed under these conditions.

# Preparation
First we have to download all datasets.
The Benchmark dataset and BETA dataset can be downloaded from http://bci.med.tsinghua.edu.cn/download.html.



# Training and evaluating the models

# Hyperparameters
The proposed DNN is initialized by sampling the weights from the Gaussian distribution with 0 mean and 0.01 variance, except that all of the weights in the first layer are initialized with 1's. We train the network in each iteration based on the training batch data , by minimizing the categorical cross entropy loss via the Adam optimizer, with the learning rate 0.0001 (with no decaying). We also incorporate drop-outs between the second and third, third and fourth, and fourth and fifth layers with probabilities 0.1, 0.1, and 0.95, respectively. 

We train the network in two stages. The first stage takes a global perspective by training with all of the data (ones in the training set) whereas the second stage re-initializes the network with the global model and fine-tunes it to each subject separately by training with only the corresponding subject data (of the training set). Hence, in the end, each subject has its own model as well. We note that, except a few, most of the existing studies do either develop only a local model or only a global model, which indicates that our introduced two-stage training is a novel contribution to BCI SSVEP spellers. We have observed that this idea of transfer learning with two-staged learning, since it takes into account the inter-subject statistical variations, provides significant ITR improvements.
