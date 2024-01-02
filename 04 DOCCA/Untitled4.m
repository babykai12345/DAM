load 'name.mat'
X = xlsread('./t1_gene_label0.xlsx');
Y = xlsread('./t1_roi_label0.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';

% 第一时期、标签0的热图
figure(1) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %获取句柄
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %获取句柄
h.XTickLabelRotation=90;
colorbar;

X = xlsread('./t1_gene_label1.xlsx');
Y = xlsread('./t1_roi_label1.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% 第一时期、标签1的热图
figure(2) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %获取句柄
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %获取句柄
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t1_gene_label2.xlsx');
Y = xlsread('./t1_roi_label2.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% 第一时期、标签2的热图
figure(3) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %获取句柄
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %获取句柄
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t1_gene_label3.xlsx');
Y = xlsread('./t1_roi_label3.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% 第一时期、标签3的热图
figure(4) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %获取句柄
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %获取句柄
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t4_gene_label0.xlsx');
Y = xlsread('./t4_roi_label0.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% 第四时期、标签0的热图
figure(5) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %获取句柄
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %获取句柄
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t4_gene_label1.xlsx');
Y = xlsread('./t4_roi_label1.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% 第四时期、标签1的热图
figure(6) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %获取句柄
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %获取句柄
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t4_gene_label2.xlsx');
Y = xlsread('./t4_roi_label2.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% 第四时期、标签2的热图
figure(7) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %获取句柄
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %获取句柄
h.XTickLabelRotation=90;
colorbar;



X = xlsread('./t4_gene_label3.xlsx');
Y = xlsread('./t4_roi_label3.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% 第四时期、标签3的热图
figure(8) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %获取句柄
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %获取句柄
h.XTickLabelRotation=90;
colorbar;