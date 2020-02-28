h = 1/(N);
interp_points=zeros(N^2,2);
for(i=1:N)
    for(j=1:N)
        for(z=1:N)
        interp_points((i-1)*N+N^2*(z-1)+j,1)=h*(i-1)+h/2;
        interp_points((i-1)*N+N^2*(z-1)+j,2)=h*(j-1)+h/2;
        interp_points((i-1)*N+N^2*(z-1)+j,3)=h*(z-1)+h/2;
        end
    end
end
S = kerneleval1(interp_points);
kernel_tilde=zeros(2*N,2*N,2*N);
for i = 0:N-1
    for j =0:N-1
        for z=0:N-1
        tmp = sqrt((h/2 -(h/2+(i)*h))^2 +(h/2 -(h/2+(j)*h) )^2+ (h/2 -(h/2+(z)*h) )^2);
        tmp = 1/(1+tmp^2);
        kernel_tilde((N + i)+1 , (N + j)+1,(N + z)+1) = tmp;
        kernel_tilde((N - i)+1 , (N + j)+1,(N + z)+1) = tmp;
        kernel_tilde((N + i)+1 , (N - j)+1,(N + z)+1) = tmp;
        kernel_tilde((N - i)+1 , (N - j)+1,(N + z)+1) = tmp;
        kernel_tilde((N + i)+1 , (N + j)+1,(N + z)+1) = tmp;
        kernel_tilde((N - i)+1 , (N + j)+1,(N + z)+1) = tmp;
        kernel_tilde((N + i)+1 , (N - j)+1,(N + z)+1) = tmp;
        kernel_tilde((N - i)+1 , (N - j)+1,(N + z)+1) = tmp;
        
        kernel_tilde((N + i)+1 , (N + j)+1,(N - z)+1) = tmp;
        kernel_tilde((N - i)+1 , (N + j)+1,(N - z)+1) = tmp;
        kernel_tilde((N + i)+1 , (N - j)+1,(N - z)+1) = tmp;
        kernel_tilde((N - i)+1 , (N - j)+1,(N - z)+1) = tmp;
        kernel_tilde((N + i)+1 , (N + j)+1,(N - z)+1) = tmp;
        kernel_tilde((N - i)+1 , (N + j)+1,(N - z)+1) = tmp;
        kernel_tilde((N + i)+1 , (N - j)+1,(N - z)+1) = tmp;
        kernel_tilde((N - i)+1 , (N - j)+1,(N - z)+1) = tmp;
        
        
        end
    end
end


w=[1:N^3]';
b=S*w;
fa=zeros(2*N,2*N,2*N);
for(i=1:N)
    for(j=1:N)
        for(z=1:N)
            fa(i,j,z)=w((i-1)*N+j+(z-1)*N^2);
        end
    end
end

result=ifftn(fftn(fa).*fftn(kernel_tilde));
result= (result(N+1:2*N,N+1:2*N,N+1:2*N));
final=zeros(N^3,1);
for(i=1:N)
    for(j=1:N)
        for(z=1:N)
            final((i-1)*N+(z-1)*N^2+j)=result(i,j,z);
        end
    end
end

