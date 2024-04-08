function y = soft(x,lambda)
n = size(x,1);
temp = sort(abs(x),'descend');
th = temp(lambda,:);
y = sign(x).*max(abs(x)-repmat(th,n,1),0);
ny = sqrt(sum(y.^2));
y = y./repmat(ny,n,1);
end