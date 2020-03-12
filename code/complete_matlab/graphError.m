function [error] = graphError(ydata,Nmax,Nmin,ks)
sum_ydata = sum(ydata .^ 2, 2);
n=length(ydata);
num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
num(1:n+1:end) = 0;                                                 % set diagonal to zero
Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
QQ=Q.*num;
realRep=4*(diag(sum(QQ, 1)) - QQ) * ydata;
error=zeros(Nmax,2);
widthy=max(ydata(:,1));
widthy2=min(ydata(:,1));
widthx=max(ydata(:,2));
widthx2=min(ydata(:,2));
figure;
j=1;
s=size(ydata,2);

for k=ks
    for N=Nmin:Nmax
        if s==1
            rep=4*gradFft1D(ydata,n,N,k);
        elseif s==2
            rep=4*gradFft2D(ydata,n,N,k);
        elseif s==3
            rep=4*gradFft3D(ydata,n,N,k);
        else
            rep=4*gradFft4D(ydata,n,N,k);
        end
        error(N,j)=norm(realRep-rep);
        error(N,j)=error(N,j)/norm(realRep);
        error(N,j)=log10(error(N,j));
    end
    plot(Nmin:Nmax,error(Nmin:Nmax,j),'linewidth',4);
    hold on
    j=j+1;
end
labs=[];
for k=ks
    labs=[labs string(k)];
end
xlabel('#Intervals');
ylabel("log10(RSE)");
title(n+"points in "+"["+widthy2+","+widthy+"]x["+widthx2+","+widthx+"]" );
legend(labs);
end

