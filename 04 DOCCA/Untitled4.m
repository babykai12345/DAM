load 'name.mat'
X = xlsread('./t1_gene_label0.xlsx');
Y = xlsread('./t1_roi_label0.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';

% ��һʱ�ڡ���ǩ0����ͼ
figure(1) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %��ȡ���
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %��ȡ���
h.XTickLabelRotation=90;
colorbar;

X = xlsread('./t1_gene_label1.xlsx');
Y = xlsread('./t1_roi_label1.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% ��һʱ�ڡ���ǩ1����ͼ
figure(2) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %��ȡ���
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %��ȡ���
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t1_gene_label2.xlsx');
Y = xlsread('./t1_roi_label2.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% ��һʱ�ڡ���ǩ2����ͼ
figure(3) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %��ȡ���
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %��ȡ���
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t1_gene_label3.xlsx');
Y = xlsread('./t1_roi_label3.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% ��һʱ�ڡ���ǩ3����ͼ
figure(4) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %��ȡ���
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %��ȡ���
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t4_gene_label0.xlsx');
Y = xlsread('./t4_roi_label0.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% ����ʱ�ڡ���ǩ0����ͼ
figure(5) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %��ȡ���
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %��ȡ���
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t4_gene_label1.xlsx');
Y = xlsread('./t4_roi_label1.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% ����ʱ�ڡ���ǩ1����ͼ
figure(6) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %��ȡ���
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %��ȡ���
h.XTickLabelRotation=90;
colorbar;


X = xlsread('./t4_gene_label2.xlsx');
Y = xlsread('./t4_roi_label2.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% ����ʱ�ڡ���ǩ2����ͼ
figure(7) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %��ȡ���
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %��ȡ���
h.XTickLabelRotation=90;
colorbar;



X = xlsread('./t4_gene_label3.xlsx');
Y = xlsread('./t4_roi_label3.xlsx');
Penalty1 = corr(X, Y);
Penalty1 = Penalty1';
% ����ʱ�ڡ���ǩ3����ͼ
figure(8) 
colormap('Jet');
imagesc(Penalty1,[-1,1]);
set(gca, 'XTick', 1 : 20, 'XTickLabel', gene_name,'fontsize', 10);
h=gca;  %��ȡ���
set(gca, 'YTick', 1 : 3,'YTickLabel',roi_name)
set(gca, 'TickLength', [0 0]);
h=gca;  %��ȡ���
h.XTickLabelRotation=90;
colorbar;