function [Co_module,Co_module_index,Co_module_classification,Co_module_name,errorx] =Comodule_selection(XX1,W, H1, tt0, tt1)
%
% INPUT
% W         : common basis matrix
% H1        : coefficient matrix
% H2        : coefficient matrix
% H3        : coefficient matrix
% tt        : a given threshold for z-score.
% OUTPUT
% Co_module : the index list of WSI, DNA methylation, CNV
%
% Compute the mean(meadia) and std in columns of W and rows in H1, H2, H3 to determine
% the module member and output the Co-module based on W and H1, H2, H3
% matrices.
% %
X1 = csvread('../02 JDSNMF/train_x_gene_noname.csv');
% X1 = normalize(X1, 'minmax');
% X2 = normalize(X2, 'minmax');

% load './name_meth2gene.mat'
load './name.mat'
m1 = size(H1,2);
n = size(W,1);
K = size(W,2);

MW = mean(W,1);     MH1 = mean(H1,2);
VW = std(W,0,1);    VH1 = std(H1,0,2);
% Co-Module
    for i = 1:K
        c1_index = find(H1(i,:) > MH1(i) + tt1*VH1(i));
        r_index=find(W(find(W(:,i))));
        c1 = H1(find(H1(i,:) > MH1(i) + tt1*VH1(i)));
        r = W(find(W(:,i)));
        Co_module{i,1}=r';Co_module{i,2}=c1;
        Co_module_index{i,1}=r_index;Co_module_index{i,2}=c1_index';
        Co_module_name{i,1}=gene_name(c1_index,:); 
%        Co_module_name{i,2}=meth2gene(c2_index);
        Co_module_classification{i,1}=X1(:,c1_index);
   score{i,1}= Co_module{i,1}'*Co_module{i,2};

%    if i==31
%     XXX1 = reshape(X1(r_index,c1_index), length(r_index)*length(c1_index), 1);
%     VVV1 = reshape(Co_module{i,1}'*Co_module{i,2} , length(r_index)*length(c1_index), 1);
%     P1 = corr(XXX1,VVV1,'type','Pearson');
%    end
        errorx1 = mean(mean(abs(XX1(r_index,c1_index)-Co_module{i,1}'*Co_module{i,2})))/mean(mean(XX1));
        errorx{i,1} = errorx1;
%         score1{i,1} = corr(XXX1, VV1, 'type', 'Spearman');
%         score1{i,2} = corr(XXX2, VV2, 'type', 'Spearman');
%         score1{i,3} = (abs(score1{i,1})+abs(score1{i,2}))/2;

   end
    