%% Block toeplitz matrix vector multiply.
clear all;
S =[ 1.0000    0.9000    0.6923    0.9000    0.8182    0.6429    0.6923    0.6429    0.5294
    0.9000    1.0000    0.9000    0.8182    0.9000    0.8182    0.6429    0.6923    0.6429
    0.6923    0.9000    1.0000    0.6429    0.8182    0.9000    0.5294    0.6429    0.6923
    0.9000    0.8182    0.6429    1.0000    0.9000    0.6923    0.9000    0.8182    0.6429
    0.8182    0.9000    0.8182    0.9000    1.0000    0.9000    0.8182    0.9000    0.8182
    0.6429    0.8182    0.9000    0.6923    0.9000    1.0000    0.6429    0.8182    0.9000
    0.6923    0.6429    0.5294    0.9000    0.8182    0.6429    1.0000    0.9000    0.6923
    0.6429    0.6923    0.6429    0.8182    0.9000    0.8182    0.9000    1.0000    0.9000
    0.5294    0.6429    0.6923    0.6429    0.8182    0.9000    0.6923    0.9000    1.0000];
u=1:9;
n=9;
m=3;
a=[];
for(i=1:m)
    a=[a S((n-(i-1)*m):-1:(n-(i)*m+1),1)'];
    a=[a S((n-(i)*m+1),2:m)];
end

for(i=1:m-1)
    a=[a S((m:-1:1),i*m+1)'];
    a=[a S(1,i*m+2:(i+1)*m)];
    
end

v=zeros(1,2*m*(m-1)+1);
for(i=1:length(u) )
    i1=ceil(i/m);
    
    i2=mod(i,m);
    if(mod(i,m)==0)
        i2=m;
    end
    v(2*m*(m-1)-(i1-1)*(2*m-1)-(i2-1)+1)=u(i);
    
end

b=a(length(a):-1:1);
p=v(length(v):-1:1);
c=conv(b,p); %% can be done with FFT

d=zeros(1,9);
for i1=1:m
    for i2=1:m
        z=2*m*(2*m-1)-i1*(2*m-1)-i2+1;
        d(10-((i1-1)*m+i2))=c(z);
    end
end
error=norm(d'-S*u');


