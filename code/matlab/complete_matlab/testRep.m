clear;
%close all;
n=1000;
width=40;
Y=(rand(n,4)-1/2)*width;
ydata=Y(:,1:3);
sum_ydata = sum(ydata .^ 2, 2);
num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
num(1:n+1:end) = 0;                                                 % set diagonal to zero
Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
QQ=Q.*num;
realRep=4*(diag(sum(QQ, 1)) - QQ) * ydata;
error=zeros(80,3);

figure;
j=1;
for k=[3 5]
    for N=10:20
        rep=4*gradFft3D(ydata,n,N,k);
        error(N,j)=norm(realRep-rep);
        error(N,j)=error(N,j)/norm(realRep);
        error(N,j)=log10(error(N,j));
    end
    plot(10:20,error(10:20,j),'linewidth',4);
    hold on
    j=j+1;
end
xlabel('#Intervals');
ylabel("log10(RSE)");
title(n+" Random points in "+"[-"+width/2+","+width/2+"]^3" )
legend("3", "5");