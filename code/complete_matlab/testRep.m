clear;
close all;
n=1000;
Y=rand(n,4)*100;
ydata=Y(:,1:2);
sum_ydata = sum(ydata .^ 2, 2);
num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
num(1:n+1:end) = 0;                                                 % set diagonal to zero
Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
QQ=Q.*num;
realRep=4*(diag(sum(QQ, 1)) - QQ) * ydata;
error=zeros(50,3);

figure;
j=1;
for k=[5 10 15]
for N=30:80
    rep=4*gradFft2D(ydata,n,N,k);
    error(N,j)=norm(realRep-rep);
    error(N,j)=error(N,j)/norm(realRep);
    %error(N,j)=log10(error(N,j));
end
plot(error(40:end,j));
hold on
j=j+1;
end
