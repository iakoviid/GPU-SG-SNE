clear;
n=10000;
k=5;
Nint=600;
xmin=-50;
xmax=50;
x=(rand(n,1)-0.5)*(xmax-xmin);
charges=rand(n,5);

%% Brute force
distmatrix = squareform(pdist(x));
kernel = 1./(1+distmatrix.^2);
Real=kernel*charges;

%% 60
aprox=compute1D(x,charges,60,5,0,n,5);

rsre=norm(Real-aprox)/norm(Real);
rme=norm(Real-aprox,1)/norm(Real,1);
rinfe=norm(Real-aprox,inf)/norm(Real,inf);

sre=norm(Real-aprox);
me=norm(Real-aprox,1);
infe=norm(Real-aprox,inf);
rme=norm(Real(1:100)-aprox(1:100),1)/norm(Real(1:100),1);

%% 600
aprox=compute1D(x,charges,600,5,0,n,5);

rsre=norm(Real-aprox)/norm(Real);
rme=norm(Real-aprox,1)/norm(Real,1);
rinfe=norm(Real-aprox,inf)/norm(Real,inf);

sre=norm(Real-aprox);
me=norm(Real-aprox,1);
infe=norm(Real-aprox,inf);
rme=norm(Real(1:100)-aprox(1:100),1)/norm(Real(1:100),1);