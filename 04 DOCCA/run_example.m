clc;clear;


clc; clear;
X = csvread('train_x_gene_noname.csv');
Y = csvread('train_x_meth_noname.csv');

% set parameters, should be tuned before running.
% As an example, we fix them
opts.alpha1 = 1;
opts.alpha2 = 1;
opts.lambda1 = 1;
opts.lambda2 = 0.1;
opts.beta1 = 1;
opts.beta2 = 1;
opts.gamma1 = 1;
opts.gamma2 = 1;

%% training and testing
[nrow, ~] = size(X);
[test, train] = crossvalind('HoldOut', nrow, 0.7);

X_0 = X(train,:);
Y_0 = Y(train,:);
X_0 = getNormalization(X_0);
Y_0 = getNormalization(Y_0);

XX1 = corr(X');
XX2 = corr(X);
YY1 = corr(Y_0');
YY2 = corr(Y_0);

X_t = X(test,:);
Y_t = Y(test,:);
X_t = getNormalization(X_t);
Y_t = getNormalization(Y_t);

tic;
[u1, v1, obj1] = OSCCA(X_0, Y_0, opts);
tt = toc;
corr_XY1 = corr(X_t*u1,Y_t*v1);

%% results shown
% subplot(321)
% stem(u0);
% title('Ground truth: u');
% subplot(322)
% stem(v0);
% title('Ground truth: v');
% subplot(323)
% stem(u1);
% title('Estimated: u');
% subplot(324)
% stem(v1);
% title('Estimated: v');
figure
plot(obj1,'-','LineWidth',1.5);
title('Objective value');