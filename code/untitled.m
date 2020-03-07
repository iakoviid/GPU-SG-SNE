clear;
error=zeros(20,10);
n=1000;
Y=rand(n,4)*40;
%Y=Y/(max(max(Y)));
%a=[];
charges=[Y(:,1) ones(n,1)];
distmatrix = squareform(pdist(Y(:,1)));
A = 1./(1+distmatrix.^2);
A=A.^2;
f=A*charges;
for k=3:3:10
for intpar=2:20
    phi=compute1D(Y(:,1),charges,intpar,k,1,n,2);
    error(intpar,k)=sum(sum((f-phi).^2))/sum(sum((f).^2));
    %error(intpar,k)=log10(error(intpar,1));
    %a=[a intpar^2];
end
end

