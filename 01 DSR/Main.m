 %Compute canonical correlations at the outputs.

%% 
clear all
clc

load 'dataset.mat'; %Partition Dataset for the 5 fold test



for n=1:5
    
    M1_te=Xt{1,n};
    M2_te=Yt{1,n};
    M1_tr=X{1,n};
    M2_tr=Y{1,n};
 
    %%
    test_X=M1_te;
    test_Y=M2_te;
    train_X=M1_tr;
    train_Y=M2_tr; 
    gnd=Label{1,n};
    %%main regression function
    % Set parameters

paraset=[0.01 0.1 1 10];


    if n==1
    for i=1:length(paraset)
        para.lambda1 = paraset(i);
        for j=1:length(paraset)
            para.lambda2 = paraset(j);
            kfold=5;
            kk=2;     
            % construct the index of cross_validation for each task.               
            [tcv fcv]=f_myCV(gnd',kfold,kk); 
            %% begin to 5-fold.
            for cc=1:kfold 
                trLab=tcv{cc}';
                teLab=fcv{cc}';
                X_tr=train_X(trLab,:);
                Y_tr=train_Y(trLab,:);
                X_te=train_X(teLab,:);
                Y_te=train_Y(teLab,:);
           
                [w1opt,w2opt,obj] = f_sCCA1(X_tr,Y_tr,para);
               

                %%
                comp_Xopt = X_te * w1opt;
                comp_Yopt = Y_te * w2opt;
                CCopt(cc) = comp_Xopt' * comp_Yopt;
            end
            res_CC(i,j)=mean(CCopt);
            i
        end
    end
 
    tempCC=0;
    for ii=1:length(paraset)
        for jj=1:length(paraset)
           if  res_CC(ii,jj)>tempCC
             tempCC=res_CC(ii,jj);
             paraOpt=[ii,jj];
           end
        end
    end
    paraOpt
    paraFinal.lambda1=paraset(paraOpt(1));
    paraFinal.lambda2=paraset(paraOpt(2));

    end
   [w1,w2,obj] = f_sCCA1(train_X,train_Y,paraFinal);
   
    %%
    comp_Xtr = train_X * w1;
    comp_Ytr = train_Y * w2;
    CCtr(n) = comp_Xtr' * comp_Ytr;
    comp_Xte = test_X * w1;
    comp_Yte = test_Y * w2;
    CCte(n) = comp_Xte' * comp_Yte;
    Weight1{n}=w1;
    Weight2{n}=w2;
    paraCV{n}=paraOpt;
    
     
end

mean_W1 = (Weight1{1}+Weight1{2}+Weight1{3}+Weight1{4}+Weight1{5})/5;
mean_W2 = (Weight2{1}+Weight2{2}+Weight2{3}+Weight2{4}+Weight2{5})/5;
CCtr1_final = mean(CCtr);
CCte1_final = mean(CCte);



