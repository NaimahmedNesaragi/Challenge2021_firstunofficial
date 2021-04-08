function [model] = model_code(compl_feat,lab_t,num_leads)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

model={};

s_idx = find(sum(lab_t,2)==1);
normaldata = compl_feat(s_idx);
labels = lab_t(s_idx,:);
normalgroups = labels(:,22);
normalgroups2=normalgroups+1;
normalgroups2=categorical(normalgroups2);

% normal vs others
 layers = [ ...
    sequenceInputLayer(num_leads*4)
    lstmLayer(100,'OutputMode','last','InputWeightsInitializer','he')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];


 options = trainingOptions('sgdm', ...
    'MaxEpochs',250, ...
    'MiniBatchSize', 125, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'Momentum',0.9,...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

net_normal = trainNetwork(normaldata,normalgroups2,layers,options);
 model{1}=net_normal;
% other data

idx_other = find(normalgroups==0);
otherdata=normaldata(idx_other);
othergroups = labels(idx_other,:);
othergroups2=othergroups+1;
othergroups2=categorical(othergroups2);

 b1=[2 3 4 13 14 15 16 24 27];% AF Flutter Bradycardia PAC PVC PRO-PR PRO-QT 
% rythm vs others

rythmdata=otherdata;
rythmgroups=sum(othergroups(:,b1),2);
rythmgroups2=rythmgroups+1;
rythmgroups2=categorical(rythmgroups2);
rythm_model = trainNetwork(rythmdata,rythmgroups2,layers,options);
model{2}=rythm_model;

% AF vs others
af_idx=find(rythmgroups==1);
rytreelabels=othergroups(af_idx,:);
afdata=otherdata(af_idx);
afgroups=rytreelabels(:,2);
afgroups2=afgroups+1;
afgroups2=categorical(afgroups2);
af_model =  trainNetwork(afdata,afgroups2,layers,options);
model{3}=af_model;


% Flutter vs others
flutterdata=afdata;
fluttergroups=rytreelabels(:,3);
fluttergroups2=fluttergroups+1;
fluttergroups2=categorical(fluttergroups2);
flutter_model = trainNetwork(flutterdata,fluttergroups2,layers,options);
model{4}=flutter_model;

% Bradycardia vs others
bradydata=afdata;
bradygroups=rytreelabels(:,4);
bradygroups2=bradygroups+1;
bradygroups2=categorical(bradygroups2);
brady_model =  trainNetwork(bradydata,bradygroups2,layers,options);
model{5}=brady_model;

% PAC vs others
pacdata=afdata;
pacgroups=sum(rytreelabels(:,[13 24]),2);
pacgroups2=pacgroups+1;
pacgroups2=categorical(pacgroups2);
pac_model = trainNetwork(pacdata,pacgroups2,layers,options);
model{6}=pac_model;

% PVC vs others
pvcdata=afdata;
pvcgroups=sum(rytreelabels(:,[14 27]),2);
pvcgroups2=pvcgroups+1;
pvcgroups2=categorical(pvcgroups2);
pvc_model = trainNetwork(pvcdata,pvcgroups2,layers,options);
model{7}=pvc_model;

% pro-QT vs others
pqtdata=afdata;
pqtgroups=rytreelabels(:,16);
pqtgroups2=pqtgroups+1;
pqtgroups2=categorical(pqtgroups2);
pqt_model = trainNetwork(pqtdata,pqtgroups2,layers,options);
model{8}=pqt_model;

%% other branch

nonry_idx=find(rythmgroups==0);
othtreelabels=othergroups(nonry_idx,:);
nrthdata=otherdata(nonry_idx);

% 1st degree vs others
idbdata=nrthdata;
idbgroups=othtreelabels(:,1);
idbgroups2=idbgroups+1;
idbgroups2=categorical(idbgroups2);
idb_model = trainNetwork(idbdata,idbgroups2,layers,options);
model{9}=idb_model;

% CRBBB vs others
crbbbdata=nrthdata;
crbbbgroups=sum(othtreelabels(:,[5 19]),2);
crbbbgroups2=crbbbgroups+1;
crbbbgroups2=categorical(crbbbgroups2);
crbbb_model = trainNetwork(crbbbdata,crbbbgroups2,layers,options);
model{10}=crbbb_model;

% IRBBB vs others
irbbbdata=nrthdata;
irbbbgroups=othtreelabels(:,6);
irbbbgroups2=irbbbgroups+1;
irbbbgroups2=categorical(irbbbgroups2);
irbbb_model = trainNetwork(irbbbdata,irbbbgroups2,layers,options);
model{11}=irbbb_model;

% LAFB vs others
lafbdata=nrthdata;
lafbgroups=othtreelabels(:,7);
lafbgroups2=lafbgroups+1;
lafbgroups2=categorical(lafbgroups2);
lafb_model = trainNetwork(lafbdata,lafbgroups2,layers,options);
model{12}=lafb_model;

% LAD vs others
laddata=nrthdata;
ladgroups=othtreelabels(:,8);
ladgroups2=ladgroups+1;
ladgroups2=categorical(ladgroups2);
lad_model = trainNetwork(laddata,ladgroups2,layers,options);
model{13}=lad_model;

% LBBB vs others
lbbbdata=nrthdata;
lbbbgroups=othtreelabels(:,9);
lbbbgroups2=lbbbgroups+1;
lbbbgroups2=categorical(lbbbgroups2);
lbbb_model = trainNetwork(lbbbdata,lbbbgroups2,layers,options);
model{14}=lbbb_model;

% LQRS vs others
lqrsdata=nrthdata;
lqrsgroups=othtreelabels(:,10);
lqrsgroups2=lqrsgroups+1;
lqrsgroups2=categorical(lqrsgroups2);
lqrs_model = trainNetwork(lqrsdata,lqrsgroups2,layers,options);
model{15}=lqrs_model;

% NSIVCD vs others
nsivcddata=nrthdata;
nsivcdgroups=othtreelabels(:,11);
nsivcdgroups2=nsivcdgroups+1;
nsivcdgroups2=categorical(nsivcdgroups2);
nsivcd_model = trainNetwork(nsivcddata,nsivcdgroups2,layers,options);
model{16}=nsivcd_model;

% Pacing rythm vs others
prdata=nrthdata;
prgroups=othtreelabels(:,12);
prgroups2=prgroups+1;
prgroups2=categorical(prgroups2);
pr_model = trainNetwork(prdata,prgroups2,layers,options);
model{17}=pr_model;

% q wave abnormal vs others
qabdata=nrthdata;
qabgroups=othtreelabels(:,17);
qabgroups2=qabgroups+1;
qabgroups2=categorical(qabgroups2);
qab_model = trainNetwork(qabdata,qabgroups2,layers,options);
model{18}=qab_model;

% RAD vs others
raddata=nrthdata;
radgroups=othtreelabels(:,18);
radgroups2=radgroups+1;
radgroups2=categorical(radgroups2);
rad_model = trainNetwork(raddata,radgroups2,layers,options);
model{19}=rad_model;

% SA vs others
sadata=nrthdata;
sagroups=othtreelabels(:,20);
sagroups2=sagroups+1;
sagroups2=categorical(sagroups2);
sa_model = trainNetwork(sadata,sagroups2,layers,options);
model{20}=sa_model;

% SB vs others
sbdata=nrthdata;
sbgroups=othtreelabels(:,21);
sbgroups2=sbgroups+1;
sbgroups2=categorical(sbgroups2);
sb_model = trainNetwork(sbdata,sbgroups2,layers,options);
model{21}=sb_model;

% STach vs others
stdata=nrthdata;
stgroups=othtreelabels(:,23);
stgroups2=stgroups+1;
stgroups2=categorical(stgroups2);
st_model = trainNetwork(stdata,stgroups2,layers,options);
model{22}=st_model;

% t wave abnormal vs others
tabdata=nrthdata;
tabgroups=othtreelabels(:,25);
tabgroups2=tabgroups+1;
tabgroups2=categorical(tabgroups2);
tab_model = trainNetwork(tabdata,tabgroups2,layers,options);
model{23}=tab_model;

% t wave inversion vs others
tinvdata=nrthdata;
tinvgroups=othtreelabels(:,26);
tinvgroups2=tinvgroups+1;
tinvgroups2=categorical(tinvgroups2);
tinv_model = trainNetwork(tinvdata,tinvgroups2,layers,options);
model{24}=tinv_model;


end

