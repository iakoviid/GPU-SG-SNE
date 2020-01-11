b=zeros(4,2);
for(i=1:N)
for(j=1:N)
b((i-1)*N+j,1)=interp_points(i,j,1);
b((i-1)*N+j,2)=interp_points(i,j,2);
end
end

for(i=1:N^2)
    for(j=1:N^2)
        v=zeros(2,1);
        u=zeros(2,1);
        v(1)=b(i,1);
        v(2)=b(i,2);
        
        u(1)=b(j,1);
        u(2)=b(j,2);
        kernel(i,j)=1/(1+norm(u-v))^2;
    end
end
kernel_tilde=zeros(2*N,2*N);
for(i=1:N) 
   for (j=1:N) 
        tmp = kernel(i,j);
        kernel_tilde(N + (i- 1)+1, N + (j-1)+1) = tmp;
        kernel_tilde(N - (i -1)+1, N + (j-1)+1) = tmp;
        kernel_tilde(N + (i -1)+1, N - (j-1)+1) = tmp;
        kernel_tilde(N - (i -1)+1, N - (j-1)+1) = tmp;
        kernel_tilde
end
end