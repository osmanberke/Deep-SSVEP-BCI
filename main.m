% Training and testing of the proposed deep neural network (DNN) in [1]  
% for steady-state visual evoked potentials SSVEP-based BCI  

% [1] Osman Berke Guney, Muhtasham Oblokulov, and Huseyin Ozkan, 
% “A Deep Neural Network for SSVEP-based Brain Computer Interfaces,” 
% arXiv, 2020.

%% Preliminaries
% Please download benchmark [2] and/or BETA [3] datasets 
% and add folder that contains downloaded files to the MATLAB path.

% [2] Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
% ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and 
% Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.

% [3] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
% benchmark database toward ssvep-bci application,” Frontiers in
% Neuroscience, vol. 14, p. 627, 2020.
%% Specifications (e.g. number of character) of datasets
subban_no=3; % # of subbands/bandpass filters
dataset='Bench'; % 'Bench' or 'BETA' dataset
signal_length=0.4; % Signal length in second
if strcmp(dataset,'Bench')
    totalsubject=35; % # of subjects
    totalblock=6; % # of blocks
    totalcharacter=40; % # of characters
    sampling_rate=250; % Sampling rate
    visual_latency=0.14; % Average visual latency of subjects
    visual_cue=0.5; % Length of visual cue used at collection of the dataset
    sample_length=sampling_rate*signal_length; % Sample length     
    total_ch=64; % # of channels used at collection of the dataset  
    max_epochs=1000; % # of epochs for first stage
    dropout_second_stage=0.6; % Dropout probabilities of first two dropout layers at second stage
elseif strcmp(dataset,'BETA')
    totalsubject=70;
    totalblock=4;
    totalcharacter=40;
    sampling_rate=250;
    visual_latency=0.13;
    visual_cue=0.5;
    sample_length=sampling_rate*signal_length; %
    total_ch=64;
    max_epochs=800;
    dropout_second_stage=0.7;
    %else %if you want to use another dataset please specify parameters of the dataset 
    % totalsubject= ... ,
    % totalblock= ... ,
    % ...
end

%% Preprocessing 
total_delay=visual_latency+visual_cue; % Total undesired signal length in seconds
delay_sample_point=round(total_delay*sampling_rate); % # of data points correspond for undesired signal length
sample_interval = (delay_sample_point+1):delay_sample_point+sample_length; % Extract desired signal
channels=[48 54 55 56 57 58 61 62 63];% Indexes of 9 channels: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
% To use all the channels set channels to 1:total_ch=64;
[AllData,y_AllData]=PreProcess(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset);
% Dimension of AllData:
% (# of channels, # sample length, #subbands, # of characters, # of blocks, # of subjects)
%% Evaluations
acc_matrix=zeros(totalsubject,totalblock); % Initialization of accuracy matrix
sizes=size(AllData);
% Leave-one-block-out, each block is used as testing block in return,
% remaining blocks are used for training of the DNN.
for block=1:totalblock
    allblock=1:totalblock;
    allblock(block)=[]; % Exclude the block used for testing    
    layers = [ ...
        imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none')
        convolution2dLayer([1,1],1,'WeightsInitializer','ones')
        convolution2dLayer([sizes(1),1],120,'WeightsInitializer','narrow-normal')
        dropoutLayer(0.1)
        convolution2dLayer([1,2],120,'Stride',[1,2],'WeightsInitializer','narrow-normal')
        dropoutLayer(0.1)
        reluLayer
        convolution2dLayer([1,10],120,'Padding','Same','WeightsInitializer','narrow-normal')
        dropoutLayer(0.95)
        fullyConnectedLayer(totalcharacter,'WeightsInitializer','narrow-normal')
        softmaxLayer
        classificationLayer];      
    
    train=AllData(:,:,:,:,allblock,:); %Getting training data
    train=reshape(train,[sizes(1),sizes(2),sizes(3),totalcharacter*length(allblock)*totalsubject*1]);
    
    train_y=y_AllData(:,:,allblock,:);
    train_y=reshape(train_y,[1,totalcharacter*length(allblock)*totalsubject*1]);    
    train_y=categorical(train_y);
    
    % First stage training
    options = trainingOptions('adam',... % Specify training options for first-stage training
        'InitialLearnRate',0.0001,...
        'MaxEpochs',max_epochs,...
        'MiniBatchSize',100, ...
        'Shuffle','every-epoch',...
        'L2Regularization',0.001,...
        'ExecutionEnvironment','gpu',...
        'Plots','training-progress');    
    main_net = trainNetwork(train,train_y,layers,options);    
    sv_name=['main_net_',int2str(block),'.mat']; 
    save(sv_name,'main_net'); % Save the trained model
    all_conf_matrix=zeros(40,40); % Initialization of confusion matrix 
    
    % Second stage training 
    for s=1:totalsubject       
        layers = [ ...
            imageInputLayer([sizes(1),sizes(2),sizes(3)],'Normalization','none')
            convolution2dLayer([1,1],1)
            convolution2dLayer([sizes(1),1],120)
            dropoutLayer(dropout_second_stage)
            convolution2dLayer([1,2],120,'Stride',[1,2])
            dropoutLayer(dropout_second_stage)
            reluLayer
            convolution2dLayer([1,10],120,'Padding','Same')
            dropoutLayer(0.95)
            fullyConnectedLayer(totalcharacter)
            softmaxLayer
            classificationLayer];
        % Transfer the weights that learnt in the first-stage training
        layers(2, 1).Weights = main_net.Layers(2, 1).Weights;
        layers(3, 1).Weights = main_net.Layers(3, 1).Weights;
        layers(5, 1).Weights = main_net.Layers(5, 1).Weights;
        layers(8, 1).Weights = main_net.Layers(8, 1).Weights;
        layers(10, 1).Weights = main_net.Layers(10, 1).Weights;
        
        layers(2, 1).Bias = main_net.Layers(2, 1).Bias;       
        layers(3, 1).Bias = main_net.Layers(3, 1).Bias;
        layers(5, 1).Bias = main_net.Layers(5, 1).Bias;
        layers(8, 1).Bias = main_net.Layers(8, 1).Bias;
        layers(10, 1).Bias = main_net.Layers(10, 1).Bias;       
       
        % Getting the subject-specific data
        train=AllData(:,:,:,:,allblock,s);
        train=reshape(train,[sizes(1),sizes(2),sizes(3),totalcharacter*length(allblock)*1]);
       
        train_y=y_AllData(:,:,allblock,s);
        train_y=reshape(train_y,[1,totalcharacter*length(allblock)*1]);   
        
        testdata=AllData(:,:,:,:,block,s);
        testdata=reshape(testdata,[sizes(1),sizes(2),sizes(3),totalcharacter]);
        
        test_y=y_AllData(:,:,block,s);
        test_y=reshape(test_y,[1,totalcharacter*1]);
        test_y=categorical(test_y);
        train_y=categorical(train_y);          
        
        options = trainingOptions('adam',... % Specify training options for first-stage training
            'InitialLearnRate',0.0001,...
            'MaxEpochs',1000,...
            'MiniBatchSize',totalcharacter*(totalblock-1), ...
            'Shuffle','every-epoch',...
            'L2Regularization',0.001,...
            'ExecutionEnvironment','gpu');
        net = trainNetwork(train,train_y,layers,options);
        
        [YPred,~] = classify(net,testdata);
        acc=mean(YPred==test_y');
        acc_matrix(s,block)=acc;       
        
        all_conf_matrix=all_conf_matrix+confusionmat(test_y,YPred);
    end
    sv_name=['confusion_mat_',int2str(block),'.mat'];
    save(sv_name,'all_conf_matrix');    
    
    sv_name=['acc_matrix','.mat'];
    save(sv_name,'acc_matrix');    
end

itr_matrix=itr(acc_matrix,totalcharacter,visual_cue+signal_length);