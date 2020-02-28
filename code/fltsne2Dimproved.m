clear all;
rng(1)
a = 0; b=1; n = 1000;
locs = rand(n,2)*(b-a);
distmatrix = squareform(pdist(locs));
kernel = 1./(1+distmatrix.^2);
v(:,1) = sin(10*locs(:,1)) + cos(2000*locs(:,1)); %Anything could be used here
v(:,2) = sin(10*locs(:,2)) + cos(2000*locs(:,2)); %Anything could be used here

f = kernel*v;%Result

%% Number of Nint intervals and k interpolation points per dimension make Nint^2 boxes with k^2 points inside them interval h interval
%length
k = 5;
Nint = 1;
h = 1/(Nint *k);


interp_points = zeros(k^2,Nint^2,2);
for j=1:k
    for i=1:k
        for intx=1:Nint
            for inty=1:Nint
                interp_points(j+(i-1)*k,intx+(inty-1)*Nint,1) = h/2 + ((j-1)+(intx-1)*k)*h;
                interp_points(j+(i-1)*k,intx+(inty-1)*Nint,2) = h/2 + ((i-1)+(inty-1)*k)*h;
            end
        end
    end
end
%% We need to be able to look up which box each point belongs to
int_lookup = zeros(n,1);

for i=1:n
    current_intx = 0;
    current_inty=0;
    while (k*h*(current_intx) < locs(i,1))
        current_intx = current_intx +1;
    end
    while (k*h*(current_inty) < locs(i,2))
        current_inty = current_inty +1;
    end
   int_lookup(i) = current_intx+(current_inty-1)*Nint;
end
gscatter(locs(:,1),locs(:,2),int_lookup);

%% Make V, which is now n rows by Nint*k columns
Vx = zeros(Nint*k,n);
for ti=1:k
    for yj=1:n
        current_int = int_lookup(yj);
        num = 1;
        denom = 1;
        for tii=1:k
            if (tii ~= ti)
                denom = denom*(interp_points(ti,current_int,1) -interp_points(tii,current_int,1));
                num= num*(locs(yj,1) - interp_points(tii,current_int,1));
            end
        end
        box_i = mod(current_int,Nint);

        Vx((box_i)*k+ti,yj) = num/denom;
    end
end

Vy = zeros(Nint*k,n);
for ti=1:k
    for yj=1:n
        current_int = int_lookup(yj);
        num = 1;
        denom = 1;
        for tii=1:k
            if (tii ~= ti)
                denom = denom*(floor(current_int/Nint)*k*h+h/2+ti*h -floor(current_int/Nint)*k*h+h/2+tii*h);
                num= num*(locs(yj,2) - floor(current_int/Nint)*k*h+h/2+tii*h);
            end
        end
        box_j = floor(current_int/Nint);

        Vy((box_j)*k+ti,yj) = num/denom;
    end
end
w=zeros((Nint*k)^2,2);
for i=1:n
    box_idx=int_lookup(i);
    box_j = floor(box_idx/Nint);
    box_i = mod(box_idx,Nint);
      for ( interp_i = 0:k-1) 
            for ( interp_j  = 0:k-1)  
                 idx = (box_i * k + interp_i) * (Nint * k) + (box_j * k) + interp_j;
                    w(idx+1,1)= w(idx+1,1)+Vx(interp_i+1+box_i*k,i)*Vy(interp_j+box_j*k+1,i)*v(i,1);
                    w(idx+1,2)=w(idx+1,2)+Vx(interp_i+1+box_i*k,i)*Vy(interp_j+box_j*k+1,i)*v(i,2);

            end
      end
end
N=Nint*k;
b=zeros(N^2,2);
kernel_tilde=zeros(2*N,2*N);
for i = 0:N-1
    for j =0:N-1
        tmp = sqrt((interp_points(1,1,1) -(h/2+(i)*h))^2 +(interp_points(1,1,2) -(h/2+(j)*h) )^2);
        tmp = 1/(1+tmp^2);
        kernel_tilde((N + i)+1 , (N + j)+1) = tmp;
        kernel_tilde((N - i)+1 , (N + j)+1) = tmp;
        kernel_tilde((N + i)+1 , (N - j)+1) = tmp;
        kernel_tilde((N - i)+1 , (N - j)+1) = tmp;
    end
end
fa=vec2mat(w(:,1),N);
fa=[fa zeros(N,N);zeros(N,2*N)];

result=ifft2(fft2(kernel_tilde).*fft2(fa));
result= (result(N+1:2*N,N+1:2*N))';
b(:,1)=result(:);
fa=vec2mat(w(:,2),N);
fa=[fa zeros(N,N);zeros(N,2*N)];
result=ifft2(fft2(kernel_tilde).*fft2(fa));
result= (result(N+1:2*N,N+1:2*N))';
b(:,2)=result(:);


fpol=zeros(n,2);
for i=1:n
    box_idx=int_lookup(i);
    box_j = floor(box_idx/Nint);
    box_i = mod(box_idx,Nint);
      for ( interp_i = 0:k-1) 
            for ( interp_j  = 0:k-1)  
                 idx = (box_i * k + interp_i) * (Nint * k) + (box_j * k) + interp_j;
                    fpol(i,1)= fpol(i,1)+Vx(interp_i+1+box_i*k,i)*Vy(interp_j+box_j*k+1,i)*b(idx+1,1);
                    fpol(i,2)=fpol(i,2)+Vx(interp_i+1+box_i*k,i)*Vy(interp_j+box_j*k+1,i)*b(idx+1,2);

            end
      end
end
error(N)=norm(f-fpol)/norm(f);
fprintf("N=%d error=%f\n",N,error(N));
