function [features] = get_features(data, header_data,leads_idx) %get_ECGLeads_features

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Extract features from ECG signals of every lead
% Inputs:
% 1. ECG data from available leads (data)
% 2. Header files including the number of leads (header_data)
% 3. The available leads index (in data/header file)
%
% Outputs:
% features for every ECG lead:
% 1. Age 2. Sex 3. root square mean (RSM) of the ECG leads
%
% Author: Nadi Sadr, PhD, <nadi.sadr@dbmi.emory.edu>
% Version 1.0
% Date 25-Nov-2020
% Version 2.1, 25-Jan-2021
% Version 2.2, 11-Feb-2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read number of leads, sample frequency and adc_gain from the header.
[recording,Total_time,num_leads,Fs,adc_gain,age,sex,Baseline] = extract_data_from_header(header_data);
num_leads = length(leads_idx);
 


    % ECG processing
    % Preprocessing
    LeadswGain=[];
    filt_ecg=[];
    res_ecg=[];
    ref_ecg=[];
    for i = [leads_idx{:}]
        % Apply adc_gain and remove baseline
        LeadswGain(i,:)   = (data(i,:)-Baseline(i))./adc_gain(i);
    end
    for i = [leads_idx{:}]
        filt_ecg(i,:)=BP_filter_ECG(LeadswGain(i,:),Fs);
    end
        if Fs<500
             for i = [leads_idx{:}]
               res_ecg(i,:)=resample(filt_ecg(i,:),500,Fs);
             end
             Fs=500;
             ref_ecg=ecg_noisecancellation( res_ecg, Fs);
        else
             ref_ecg=ecg_noisecancellation( filt_ecg, Fs);
        end
        if (length(ref_ecg)>10*Fs-1)
            ref_ecg2=ref_ecg(:,1:10*Fs);
        else
            for i=1:12
                ref_ecg2(i,:)=resample(ref_ecg(i,:),10*Fs,length(ref_ecg(i,:)));
            end
        end
        % QRS and P wave detection
        j=1;
        for i=1:2:23
        sequence1(i,:)=pentropy(ref_ecg2(j,:),Fs);
        sequence1(i+1,:)=instfreq(ref_ecg2(j,:),Fs);
        j=j+1;
        end
        %sequence11=reshape(sequence1,1,3096);
        sequence1(find(isnan(sequence1)))=0;
        sequence1(find(isinf(sequence1)))=0;
        mu = mean(sequence1,2);
        sg = std(sequence1,[],2);
        
        sequence2=(sequence1-mu)./sg;
        sequence2(find(isnan(sequence2)))=0;
        sequence2(find(isinf(sequence2)))=0;
        seq3=sequence2(:,1:121);
        seq4=reshape(seq3,24,11,11);
        
        t11=gather(seq4);
        t2=tensor(t11);
        t2_tuc=tucker_als(t2,[24, 11, 11]);
        featt=t2_tuc.core;
        featcore=double(reshape(featt,[24,121]));
        featcore(isnan(featcore==0));
        featcore(isinf(featcore==0));
        feat=featcore; 
        
        features=[seq3; feat];
        

        
     
        
        
        
    

end


