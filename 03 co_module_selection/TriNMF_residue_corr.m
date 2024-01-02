function [X1score,X2score,X3score, sig]=TriNMF_residue_corr(X1, X2, X3)
sss1=size(X1,1);
sss3=size(X1,2);
sss4=size(X2,2);
sss5=size(X3,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmp_x11 = zeros(sss1,1);
tmp_x21 = zeros(sss1,1); 
X1score = zeros(sss1,sss3);
X2score = zeros(sss1,sss4);
for i_x1 = 1:sss3
     for i_x2 = 1:sss4
         X1 = X1';
         X2 = X2'
         tmp_x11 = X1(i_x1,:);
         tmp_x21 = X2(i_x2,:);
         X12score(i_x1,i_x2) = corr(tmp_x11,tmp_x21);
     end
end
X12score = round(X12score,4);
X12score_abs = abs(round(X12score,4));
X12final = X12score_abs(:);
[Y,I]=sort(X12final,'descend');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmp_x12 = zeros(sss1,1);
tmp_x22 = zeros(sss1,1); 
X1score = zeros(sss1,sss3);
X3score = zeros(sss1,sss5);
for i_x1 = 1:sss3
     for i_x2 = 1:sss5
         tmp_x11 = X1(:,i_x1);
         tmp_x21 = X3(:,i_x2);
         X13score(i_x1,i_x2) = mean(xcorr(tmp_x12,tmp_x22,'none'));
     end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmp_x14 = zeros(sss1,1);
tmp_x24 = zeros(sss1,1); 
X2score = zeros(sss1,sss4);
X3score = zeros(sss1,sss5);
for i_x1 = 1:sss4
     for i_x2 = 1:sss5
         tmp_x14 = X2(:,i_x1);
         tmp_x24 = X3(:,i_x2);
         X23score(i_x1,i_x2) = mean(xcorr(tmp_x14,tmp_x24,'none'));
     end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X12score1=sum(X12score(:)');
X13score1=sum(X13score(:)');
X23score1=sum(X23score(:)');
p=1/3*[(1/(sss3*sss4))*X12score1*X12score1+(1/(sss3*sss5))*X13score1*X13score1+...
(1/(sss4*sss5))*X23score1*X23score1];
[h,sig,ci,zval] = ztest(p,0,0.2);    

end