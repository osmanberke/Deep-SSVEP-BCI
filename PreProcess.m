function [AllData,y_AllData]=PreProcess(channels,sample_length,sample_interval,subban_no,totalsubject,totalblock,totalcharacter,sampling_rate,dataset)
% Preprocessing for Deep Neural Network in 
% A Deep Neural Network for SSVEP-based Brain Computer Interfaces 
% Input: Please check main.m
% Output: -AllData: Preprocessing the data with bandpass filter/s,
%					Dimension of AllData:
%					(# of channels, # sample length, #subbands,
%					 # of characters, # of blocks, # of subjects)
%         -y_AllData: Labels of characters in AllData					

%% Initialization
total_channels=length(channels); % Determine total number of channel
AllData=zeros(total_channels,sample_length,subban_no,totalcharacter,totalblock,totalsubject); %initializing
y_AllData=zeros(1,totalcharacter,totalblock,totalsubject); %initializing

%% Forming bandpass filters
%High cut off frequencies for the bandpass filters (90 Hz for all)
high_cutoff = ones(1,subban_no)*90;
%Low cut off frequencies for the bandpass filters (ith bandpass filter low cutoff frequency 8*i)
low_cutoff =8:8:8*subban_no;
filter_order=2; % Filter Order of bandpass filters
PassBandRipple_val=1; 
bpFilters=cell(subban_no,1); % Form and store bandpass filters
for i=1:subban_no
    bpFilt1 = designfilt('bandpassiir','FilterOrder',filter_order, ...
        'PassBandFrequency1',low_cutoff(i),'PassBandFrequency2',high_cutoff(i),...
		'PassBandRipple',PassBandRipple_val,...
		'DesignMethod','cheby1','SampleRate',sampling_rate);        
    bpFilters{i}=bpFilt1;
end

%% Filtering
for subject=1:totalsubject
    nameofdata=['s',num2str(subject),'.mat'];
    data=load(nameofdata); % Loading the subject data
    data=data.data;
    if strcmp(dataset,'BETA')	
		data=data.EEG;	
	end
	% Taking data from spesified channels, and signal interval
    sub_data= data(channels,sample_interval,:,:); 
    
    for chr=1:1:totalcharacter        
        for blk=1:totalblock            
            if strcmp(dataset,'Bench')
                tmp_raw=sub_data(:,:,chr,blk);
            elseif strcmp(dataset,'BETA')
                tmp_raw=sub_data(:,:,blk,chr);
                %else
            end         
            for i=1:subban_no
                processed_signal=zeros(total_channels,sample_length); % Initialization
                for j=1:total_channels                       
                    processed_signal(j,:)=filtfilt(bpFilters{i},tmp_raw(j,:)); % Filtering raw signal with ith bandpass filter
                end                 
                AllData(:,:,i,chr,blk,subject)=processed_signal;
				y_AllData(1,chr,blk,subject)=chr;
            end
        end
    end    
end
end
