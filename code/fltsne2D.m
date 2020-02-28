%%
clear all

rng(1)
a = 0; b=1; n = 1000;
[locs,~] = sort(rand(n,2)*(b-a));
%locs=[0.3 0.3;0.1 0.7];
kernel = kerneleval1(locs);
v=zeros(n,2);
v(:,1) = sin(10*locs(:,1)) + cos(2000*locs(:,1)); %Anything could be used here
v(:,2) = sin(10*locs(:,2)) + cos(2000*locs(:,2)); %Anything could be used here
f=kernel*v;

%% Number of N interpolation points in each dimension N^2 total h interval length
error=zeros(21,1);
for N=2:21
h = 1/(N);
interp_points=zeros(N^2,2);
for(i=1:N)
    for(j=1:N)
        interp_points((i-1)*N+j,1)=h*(i-1)+h/2;
        interp_points((i-1)*N+j,2)=h*(j-1)+h/2;
        
    end
end

%% Lagrange Polynomials
k = N^2;% Number of interpolation points
Vx=interpolate_eval(locs(:,1),interp_points(1:N,2));
Vy=interpolate_eval(locs(:,2),interp_points(1:N,2));
V=zeros(k,n);
for i=0:N-1
    for j=1:N
        V(i*N+j,:)=Vx(i+1,:).*Vy(j,:);
    end
end


%%


w=V*v;
b=zeros(N^2,2);
kernel_tilde=zeros(2*N,2*N);
for i = 0:N-1
    for j =0:N-1
        tmp = sqrt((interp_points(1,1) -(h/2+(i)*h))^2 +(interp_points(1,2) -(h/2+(j)*h) )^2);
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



fpol=V'*b;
error(N)=norm(f-fpol)/norm(f);
error(N)=log10(error(N));
fprintf("N=%d log10error=%f\n",N,error(N));

end
plot(error)
title('log10(Relative Error)')

