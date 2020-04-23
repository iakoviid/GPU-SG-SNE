function [errorRation,speedup] =  testAverage(Y,charges,N1,percent,k,squared,scale,number)
N2=floor(N1*percent);
n=size(Y,1);
nsums=size(charges,2);
scale=(max(Y(:))-min(Y(:)))/(k*N2)*scale;
kernel=squareform(pdist(Y));
kernel = 1./(1+kernel.^2);
f=kernel*charges;
tic;
d=size(Y,2);
if d==1
    phi=compute1Dshift(Y,charges,N1,k,squared,n,nsums,0);
elseif d==2
    phi=compute2Dshift(Y,charges,N1,k,squared,n,nsums,0);
elseif d==3
    phi=compute3Dshift(Y,charges,N1,k,squared,n,nsums,0);
end
time1=toc;
width=max(Y(:))-min(Y(:));

tic;
phistar=zeros(size(phi));
if d==1
for i=0:number
    phistar=phistar+compute1Dshift(Y,charges,N2,k,squared,n,nsums,i*scale);
end
elseif d==2
for i=0:number
    phistar=phistar+compute2Dshift(Y,charges,N2,k,squared,n,nsums,i*scale);
end
elseif d==3
for i=0:number
    phistar=phistar+compute3Dshift(Y,charges,N2,k,squared,n,nsums,i*scale);
end
end
phistar=phistar/(number+1);
time2=toc;
%displaysouts(Y,f,phi,phistar,number,time1,time2,k,N2,width,scale)
errorRation=norm(f-phi)/norm(f-phistar);
speedup=time1/time2;
end
