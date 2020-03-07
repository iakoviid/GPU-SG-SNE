%%
rng(1)
a = 0; b=1; n = 1000;
[locs,~] = sort(rand(n,1)*(b-a));
distmatrix = squareform(pdist(locs));
kernel = 1./(1+distmatrix.^2);

v = sin(10*locs) + cos(2000*locs); %Anything could be used here

f = kernel*v;%Result

%% Number of Nint intervals k interpolation points per interval h interval
%length
relativeError=zeros(21,1);
for papa=2:21
k = papa;
Nint = 10; 
h = 1/(Nint *k);


%k interpolation points in each interval
interp_points = zeros(k,Nint);
for j=1:k
    for int=1:Nint
        interp_points(j,int) = h/2 + ((j-1)+(int-1)*k)*h;
    end
end


%% We need to be able to look up which interval each point belongs to
int_lookup = zeros(n,1);
current_int = 0;
for i=1:n
    if (k*h*(current_int) < locs(i))
        current_int = current_int +1;
    end
   int_lookup(i) = current_int;
end


%% Make V, which is now n rows by Nint*k columns
V = zeros(n,Nint*k);
for ti=1:k 
    for yj=1:n
        current_int = int_lookup(yj);
        num = 1;
        denom = 1;
        for tii=1:k
            if (tii ~= ti)
                denom = denom*(interp_points(ti,current_int) -interp_points(tii,current_int));
                num= num*(locs(yj) - interp_points(tii,current_int));
            end
        end

        V(yj,(current_int-1)*k+ti) = num/denom;
    end
end

%% Make S, which is k*Nint by k*Nint
S = ones(k*Nint,k*Nint);
for int1=1:Nint
    for i=1:k
        for int2=1:Nint
            for j=1:k    
                S((int1-1)*k+i,(int2-1)*k+j) = 1/(1+norm(interp_points(i,int1)-interp_points(j,int2))^2);
            end
        end
    end
end
%% FLT SNE
f_poly_approx=V'*v;

a=[0;S(k*Nint:-1:2,1)];
B=toeplitz(a);
C2=[S B;B S];

v2=[f_poly_approx;zeros(k*Nint,1)];

b_fft=ifft(fft(v2).*fft(C2(1,:))');
b_fft=b_fft(1:k*Nint);
f_poly_approx = V*b_fft;



relativeError(papa)=norm(f_poly_approx-f)/norm(f);
relativeError(papa)=log10(relativeError(papa));
disp(relativeError(papa));
end
plot(relativeError)
title('log10(Relative Error)-1D interval interp')
xlabel('Nintervals=points/int')
