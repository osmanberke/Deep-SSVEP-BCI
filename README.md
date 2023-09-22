# A Deep Neural Network for SSVEP-based Brain Computer Interfaces
This is the official repository for deep neural network (DNN) for SSVEP-based brain computer interfaces (BCI) described in paper [1]: https://ieeexplore.ieee.org/document/9531496 (Arxiv link: https://arxiv.org/abs/2011.08562).
This repository allows you to train and test, the proposed DNN model.

|![alt text](https://github.com/osmanberke/Deep-SSVEP-BCI/blob/main/system.png)|
|:--:| 
|A typical system set-up of a BCI SSVEP speller is illustrated. A matrix of thumbnail images of certain alphanumeric characters is visually presented to the user on the screen. Each character is contrast-modulated in time by a sinusoid of the assigned unique frequency, thereby generating a flickering effect during the T, e.g., T=1, seconds of visual presentation. For example, the character "C" flickers at 10 Hz as illustrated above. If the user wishes to spell a character and attends to the corresponding thumbnail, then the steady state brain response manifests the multi-channel SSVEP signal that is dominated in its spectrum by the harmonics of the input frequency, as also illustrated above in the case of "C". The goal is the target identification for spelling that is to predict the target character based on the received multi-channel SSVEP signal with C channels, e.g., C=9 or C=64. We propose a DNN architecture (with 4 convolutional layers and 1 fully connected layer) for the resulting multi-class classification problem. The proposed DNN strongly outperforms the state-of-the-art as well as the most recently proposed techniques uniformly across all signal durations {0.2,0.3, ... ,0.9,1.0}, but in particular delivers impressive information transfer rate (ITR) results that are **265.23 bits/min** and **196.59 bits/min** ITRs in as short as even T=0.4 seconds of stimulation with C=64 channels on the two publicly available large scale benchmark [2] and BETA [3] datasets .|


# Requisites

All models were implemented tested with MATLAB 2018B. All following steps assume that they are executed under these conditions.

# Preparation
First we have to download all datasets.
The Benchmark dataset [2] and BETA dataset [3] can be downloaded from http://bci.med.tsinghua.edu.cn/download.html.



# Training and evaluating the models
In our performance evaluations, we conducted the comparisons (following the procedure in the literature) in a leave-one-block-out fashion.
For example, we train on 5 (or 3) and test on the remaining block and repeat this process 6 (4) times in order to have exhaustively tested on each block in the case of the benchmark (or the BETA) dataset. For fairness, we take into account a 0.5 seconds gaze shift time while computing the ITR results (as it is computed in other methods). We test with the pre-determined set of 9 channels (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2) again for fair comparisons (since these are the channels that have been used in the compared methods), but we also test with all of the available 64 channels to fully demonstrate the efficacy of our DNN. 
# Hyperparameters
The proposed DNN is initialized by sampling the weights from the Gaussian distribution with 0 mean and 0.01 standard deviation, except that all of the weights in the first layer are initialized with 1's. We train the network in each iteration based on the training batch data , by minimizing the categorical cross entropy loss via the Adam optimizer, with the learning rate 0.0001 (with no decaying). We also incorporate drop-outs between the second and third, third and fourth, and fourth and fifth layers with probabilities 0.1, 0.1, and 0.95, respectively. 

We train the network in two stages. The first stage takes a global perspective by training with all of the data (ones in the training set) whereas the second stage re-initializes the network with the global model and fine-tunes it to each subject separately by training with only the corresponding subject data (of the training set). Hence, in the end, each subject has its own model as well. We note that, except a few, most of the existing studies do either develop only a local model or only a global model, which indicates that our introduced two-stage training is a novel contribution to BCI SSVEP spellers. We have observed that this idea of transfer learning with two-staged learning, since it takes into account the inter-subject statistical variations, provides significant ITR improvements.

# Results
The original results of the DNN using 9 channels for both the benchmark and BETA datasets are now available in the 'Results' folder.

# References 
1. O. B. Guney, M. Oblokulov and H. Ozkan, "A Deep Neural Network for SSVEP-Based Brain-Computer Interfaces," IEEE Transactions on Biomedical Engineering, vol. 69, no. 2, pp. 932-944,  2022.
2. Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
   ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and 
   Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.
3. B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
   benchmark database toward ssvep-bci application,” Frontiers in
   Neuroscience, vol. 14, p. 627, 2020.
