clear all
clc
%% importing data and making network parameters in an optimum state and partition Data for the 5 fold test.
% load dataset.mat
%X1 is genotypes data
%X2 is brain network phenotypes data
X1 = csvread('train_x_gene.csv');
% DNA methylation
X2 = csvread('train_x_meth.csv');
label = csvread('train_y_gene.csv');

%% Use the seed to reproduce the errors listed below.
%randseed=8409;
% Hidden activation type.
hiddentype='sigmoid';
% Architecture (hidden layer sizes) for genotypes data neural network.

NN1=[512 512 10]; 
%  Architecture (hidden layer sizes)  for brain network phenotypes data neural network.
NN2=[512 512 10];
     
% Weight decay parameter.
l2penalty=1e-4;

%% Run DCCA with SGD. No pretraining is used.
% Minibatchsize.
batchsize=500;
% Learning rate.
eta0=0.01;
% Rate in which learning rate decays over iterations.
% 1 means constant learning rate.
decay=1;
% Momentum.
momentum=0.99;
% How many passes of the data you run SGD with.
maxepoch=50;
addpath ./deepnet/

[N,D1]=size(X1); [~,D2]=size(X2);
  %% Set genotypes data architecture.
  Layersizes1=[D1 NN1];  Layertypes1={};
  for nn1=1:length(NN1)-1;
    Layertypes1=[Layertypes1, {hiddentype}];
  end
  % I choose to set the last layer to be linear.
  Layertypes1{end+1}='linear';
  %% Set brain network phenotypes data architecture.
 Layersizes2=[D2 NN2];  Layertypes2={};
  for nn2=1:length(NN2)-1;
   Layertypes2=[Layertypes2, {hiddentype}];
 end
Layertypes2{end+1}='linear';
  %% Random initialization of weights.
  F1=deepnetinit(Layersizes1,Layertypes1);
  F2=deepnetinit(Layersizes2,Layertypes2);
  

 
  for j=1:length(F1)  F1{j}.l=l2penalty;  end
  for j=1:length(F2)  F2{j}.l=l2penalty;  end
  
  %% the outputs at the top layer.
  FX1=deepnetfwd(X1,F1); FX2=deepnetfwd(X2,F2);

%the self-representation matrix is learned for reconstructing the source data at the top layer.
  
    options.lambda = 1;
            opts = [];
            opts.init = 0;
            opts.tFlag = 10; opts.maxIter = 1;
            opts.rFlag = 10^-5;
            opts.rsL2 = 0;
            options.opts = opts;
            options.label = label;
% 

          wSNPdata= f_SR(FX1', options);
          wSNPdata=wSNPdata+wSNPdata';
     
        
          wBNdata = f_SR(FX2', options);
            wBNdata=wBNdata+wBNdata';
           
         
            sSNPdata= wSNPdata*X1;
            sBNdata= wBNdata*X2;
            
kk=1; 
kfold = 5;
[tcv,fcv]=f_myCV(label',kfold,kk);
for cc = 1:kfold
    trLab=tcv{cc}';
    teLab=fcv{cc}';
    X{1,cc}=sSNPdata(trLab,:);  
    Y{1,cc}=sBNdata(trLab,:);
    Label{1,cc}=label(trLab,:);
    
    Xt{1,cc}=sSNPdata(teLab,:);
    Yt{1,cc}=sBNdata(teLab,:);
    Labelt{1,cc}=label(teLab,:);
    
end
csvwrite('train_x_gene_noname_re.csv', sSNPdata)
csvwrite('train_x_meth_noname_re.csv', sBNdata)


