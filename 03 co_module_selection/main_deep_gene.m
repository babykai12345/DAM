%% load the real input data
clear all
clc
rng(8049)
% load './name_meth2gene.mat'
load './name.mat'
X1 = csvread('../02 JDSNMF/train_x_gene_noname_re.csv');
X2 = csvread('../02 JDSNMF/train_x_meth_noname_re.csv');
% W = csvread('D:\Joint-deep-semi-NMF-master\U_L9.csv');
% H1 = csvread('D:\Joint-deep-semi-NMF-master\H10_L9.csv');
% H2 = csvread('D:\Joint-deep-semi-NMF-master\H20_L9.csv');
% H3 = csvread('D:\Joint-deep-semi-NMF-master\H30_L9.csv');

W = csvread('../02 JDSNMF/U_gene.csv');
H1 = csvread('../02 JDSNMF/H_gene.csv');

% W = csvread('D:\WQQLG\final_results_ours\U_L8.csv');
% H1 = csvread('D:\WQQLG\final_results_ours\H10_L8.csv');
% H2 = csvread('D:\WQQLG\final_results_ours\H20_L8.csv');
% H3 = csvread('D:\WQQLG\final_results_ours\H30_L8.csv');


[n,m1] = size(X1);
XX11 = reshape(X1, n*m1, 1);
VV1 = reshape(W*H1, n*m1, 1);
corr1 = corr(XX11, VV1, 'type', 'Pearson');
tt0 = 2; tt1 = 2;
[Co_module,Co_module_index,Co_module_classification,Co_module_name,score] =Comodule_selection(X1, W, H1, tt0, tt1);
save Comodule_final_gene.mat Co_module;
%% module elements extraction
% BB1=zeros(15,1500);
% BB2=zeros(15,1500);
% BB1(BB1==0)=[];
% BB2(BB2==0)=[];
% for i=1:K
%     AA1=Co_module(i,1);
%     AA2=Co_module(i,2);
%     AA11=cell2mat(AA1);
%     AA22=cell2mat(AA2);
%     k1=length(AA11);
%     k2=length(AA22);
%     for j1=1:k1
%         BB1(i,j1)=AA11(1,j1);
%     end
%     for j2=1:k2
%         BB2(i,j2)=AA22(1,j2);
%     end
% end
% BB1(BB1==0)=NaN;
% BB2(BB2==0)=NaN;
% corr_mean = (corr1 + corr2)/2;
% diff_step = sum(sum(abs(X1-W*H1)))
% miRNA_name_64 = gene_name(Co_module_index{3, 2}, :);
% 
% XX1_selected = X1(:, Co_module_index{3, 2});
% 
% for i = 1:151
%     for j = 1:2
%         Co_module_static{i,j} = length(Co_module_name{i,j});
%     end
% end
% 
% c = zeros(1, 154);
% XX1 = W*H1;
% for i=1:154
%     c(i)=corr(X1(i, :),XX1(i,:), 'type', 'Pearson');
% end
    