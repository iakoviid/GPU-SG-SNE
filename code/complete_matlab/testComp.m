clear;
error=zeros(20,4);
n=1000;
Y=rand(n,4)*1000+ rand(n,4)*50;
Y=(Y)/max(max(Y));
%Y=Y-0.5;
a=[];
for intpar=2:20
    charges=[Y(:,1) ones(n,1)];
    phi=compute1D(Y(:,1),charges,intpar,intpar,1,n,2);
    distmatrix = squareform(pdist(Y(:,1)));
    A = 1./(1+distmatrix.^2);
    A=A.^2;
    f=A*charges;
    error(intpar,1)=norm(f-phi)/norm(f);
    error(intpar,1)=log10(error(intpar,1));
    a=[a intpar^2];
end

for intpar=2:20
    charges=[ones(n,1) Y(:,1:2) ];
    phi=compute2D(Y(:,1:2),charges,intpar,intpar,1,n,3);
    distmatrix = squareform(pdist(Y(:,1:2)));
    A = 1./(1+distmatrix.^2);
    A=A.^2;
    f=A*charges;
    error(intpar,2)=norm(f-phi)/norm(f);
    error(intpar,2)=log10(error(intpar,2));

end
figure();
plot(a,error(2:end,1))
hold on
plot(a,error(2:end,2))

for intpar=2:11
    charges=[Y(:,1:3) ones(n,1)];
    phi=compute3D(Y(:,1:3),charges,intpar,intpar,1,n,4);
    distmatrix = squareform(pdist(Y(:,1:3)));
    A = 1./(1+distmatrix.^2);
    A=A.^2;
    f=A*charges;
    error(intpar,3)=norm(f-phi)/norm(f);
    error(intpar,3)=log10(error(intpar,3));


end

for intpar=2:5
    charges=[Y(:,1:4) ones(n,1)];
    phi=compute4D(Y(:,1:4),charges,intpar,intpar,1,n,5);
    distmatrix = squareform(pdist(Y(:,1:4)));
    A = 1./(1+distmatrix.^2);
    A=A.^2;
    f=A*charges;
    error(intpar,4)=norm(f-phi)/norm(f);
    error(intpar,4)=log10(error(intpar,4));

end

hold on
plot(a,error(2:end,3))
hold on
plot(a,error(2:end,4))

