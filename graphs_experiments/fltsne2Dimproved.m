Nint=5;
p=2;
N_int2=Nint*Nint;
h=1/Nint;
for(i=1:Nint)
    for(j=1:Nint)

        Bxlb(i*Nint+j+1)=j*h;
        Bxub(i*Nint+j+1)=(j+1)*h;
        Bylb(i*Nint+j+1)=i*h;
        Byub(i*Nint +j+1)=(i+1)*h;

    end
end

t=1/p;
y[1]=t/2;
for(i=2:p)
    y(i)=y(i-1)+t;
end

t=1/p;
for(i=1:p)
    x(i)=(i-1)*t+t/2;
end

N1d=Nint*p;
Nfft=2*N1D;
t=t*h;

interp_points=zeros(N1D,2);
for(i=1:N1D)
    interp_points[i]=t*(i-1)+h/2;
end







