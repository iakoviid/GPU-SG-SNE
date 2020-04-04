function [b] = g2g4D(w,y_tilde,squared,N1d,nsums)

kernel_tilde=zeros(2*N1d,2*N1d,2*N1d,2*N1d);
for i = 0:N1d-1
    for j =0:N1d-1
        for z=0:N1d-1
            for t=0:N1d-1
            tmp=kernel([y_tilde(1,1) y_tilde(1,2) y_tilde(1,3) y_tilde(1,4)],[y_tilde(i+1,1) y_tilde(j+1,2) y_tilde(z+1,3) y_tilde(t+1,4)],squared);
            for signi=-1:2:1
                for signj=-1:2:1
                    for signz=-1:2:1
                       for signt=-1:2:1
                        kernel_tilde((N1d +signi*i)+1 , (N1d + signj*j)+1,(N1d + signz*z)+1,(N1d + signt*t)+1 ) = tmp;
                    end
                end
                end
            end
            end
        end
    end
end

fft_kernel=fftn(kernel_tilde);

b=zeros(N1d^4,nsums);
for nterms=1:nsums
    fa=zeros(2*N1d,2*N1d,2*N1d,2*N1d);
    for(i=1:N1d)
        for(j=1:N1d)
            for(z=1:N1d)
                for(t=1:N1d)

                fa(i+N1d,j+N1d,z+N1d,t+N1d)=w((i-1)*N1d+j+(z-1)*N1d^2+(t-1)*N1d^3,nterms);
                end
            end
        end
    end
    result=ifftn(fftn(fa).*fft_kernel);
    
    result= result(1:N1d,1:N1d,1:N1d,1:N1d);
    for(i=1:N1d)
        for(j=1:N1d)
            for(z=1:N1d)
                for(t=1:N1d)
                b((i-1)*N1d+(z-1)*N1d^2+j+(t-1)*N1d^3,nterms)=result(i,j,z,t);
                end
            end
        end
    end
    
end

end

