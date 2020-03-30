clear;
error=zeros(40,4);
n=1000;
Y=rand(n,4)*40;
%Y=Y/(max(max(Y)));
a=[];
charges=[Y(:,1) ones(n,1)];
distmatrix = squareform(pdist(Y(:,1)));
A = 1./(1+distmatrix.^2);
A=A.^2;
f=A*charges;

for intpar=2:40
    phi=compute1D(Y(:,1),charges,intpar,10,1,n,2);
    error(intpar,1)=sum(sum((f-phi).^2))/sum(sum((f).^2));
    error(intpar,1)=log10(error(intpar,1));
    a=[a intpar^2];
end
charges=[ones(n,1) Y(:,1:2) ];
distmatrix = squareform(pdist(Y(:,1:2)));
A = 1./(1+distmatrix.^2);
A=A.^2;
f=A*charges;

for intpar=2:40
    phi=compute2D(Y(:,1:2),charges,intpar,10,1,n,3);
    error(intpar,2)=norm(f-phi)/norm(f);
    error(intpar,2)=log10(error(intpar,2));
    
end
%figure();
%plot(a,error(2:end,1))
%hold on
%plot(a,error(2:end,2))

charges=[Y(:,1:3) ones(n,1)];
distmatrix = squareform(pdist(Y(:,1:3)));
A = 1./(1+distmatrix.^2);
A=A.^2;
f=A*charges;
for intpar=2:20
    phi=compute3D(Y(:,1:3),charges,intpar,5,1,n,4);
    error(intpar,3)=norm(f-phi)/norm(f);
    error(intpar,3)=log10(error(intpar,3));
end

charges=[Y(:,1:4) ones(n,1)];
distmatrix = squareform(pdist(Y(:,1:4)));
A = 1./(1+distmatrix.^2);
A=A.^2;
f=A*charges;
for intpar=2:7
    
    phi=compute4D(Y(:,1:4),charges,intpar,3,1,n,5);
    error(intpar,4)=norm(f-phi)/norm(f);
    error(intpar,4)=log10(error(intpar,4));
    
end

%hold on
%plot(a,error(2:end,3))
%hold on
%plot(a,error(2:end,4))

