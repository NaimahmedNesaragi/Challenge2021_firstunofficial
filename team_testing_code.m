%% Apply classifier model to test set

function [score, label,classes] = team_testing_code(data,header_data, loaded_model)

model   = loaded_model.model;
classes = loaded_model.classes;

num_classes = length(classes);

label = zeros([1,num_classes]);

score = ones([1,num_classes]);

% Extract features from test data
tmp_hea = strsplit(header_data{1},' ');
num_leads = str2num(tmp_hea{2});
[leads, leads_idx] = get_leads(header_data,num_leads);
features1{1,1} = get_features(data,header_data,leads_idx);
if num_leads==3
Features_leads = cellfun(@(x)[x(1,:); x(2,:);x(3,:); x(4,:);x(9,:); x(10,:);x(25,:); x(26,:);x(27,:); x(28,:);x(39,:); x(40,:)],features1,'UniformOutput',false);
elseif num_leads==6
    Features_leads = cellfun(@(x)[x(1,:); x(2,:);x(3,:); x(4,:);x(5,:); x(6,:);x(7,:); x(8,:);x(9,:); x(10,:);x(11,:); x(12,:);x(25,:); x(26,:);x(27,:); x(28,:);x(29,:); x(30,:);x(31,:); x(32,:);x(33,:); x(34,:);x(35,:); x(36,:)],features1,'UniformOutput',false);
elseif num_leads==2
    Features_leads =  cellfun(@(x)[x(3,:); x(4,:);x(21,:); x(22,:);x(27,:); x(28,:);x(45,:); x(46,:)],features1,'UniformOutput',false);
else
    Features_leads=features1;   
end
% Use your classifier here to obtain a label and score for each class.
score = model_test_code(model,Features_leads,classes);
[~,idx] = find(score>0.5);

label(idx)=1;
end
