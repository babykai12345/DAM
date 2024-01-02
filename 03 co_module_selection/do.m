clear all
count = 1;
a = [];
b = [];
load Co_module_index.mat
load name.mat
for i = 1:length(Co_module_index)
         gene_list = Co_module_index(i,2);
         for j = 1:numel(gene_list)
            a = [a;cell2mat(gene_list)];
         end
end


for i = 1:length(Co_module_index)
         meth_list = Co_module_index(i,3);
         for j = 1:numel(meth_list)
            b = [b;cell2mat(meth_list)];
         end
end

gene_selected = gene_name(a);
meth_selected = meth_name(b);
