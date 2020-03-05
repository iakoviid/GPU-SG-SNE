%clear;
error=zeros(20,4);
%n=500;
%Y=rand(n,4);
for intpar=2:20
    charges=[Y(:,1) ones(n,1)];
    phi=compute1D(Y(:,1),charges,intpar,intpar,1,n,2);
    distmatrix = squareform(pdist(Y(:,1)));
    A = 1./(1+distmatrix.^2);
    A=A.^2;
    f=A*charges;
    error(intpar,1)=norm(f-phi)/norm(f);
    error(intpar,1)=log10(error(intpar,1));

end

for intpar=2:20
    charges=[Y(:,1:2) ones(n,1)];
    phi=compute2D(Y(:,1:2),charges,intpar,intpar,1,n,3);
    distmatrix = squareform(pdist(Y(:,1:2)));
    A = 1./(1+distmatrix.^2);
    A=A.^2;
    f=A*charges;
    error(intpar,2)=norm(f-phi)/norm(f);
    error(intpar,2)=log10(error(intpar,2));

end
figure();
plot(error(:,1))
hold on
plot(error(:,2))

for intpar=2:12
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
plot(error(:,3))
hold on
plot(error(:,4))

