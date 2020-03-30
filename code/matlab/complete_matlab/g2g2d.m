function [b] = g2g2d(w,N1d,x_tilde,y_tilde,squared,nsums)
kernel_tilde=zeros(2*N1d,2*N1d);
for i = 0:N1d-1
    for j =0:N1d-1

        tmp=kernel([x_tilde(1) y_tilde(1)],[x_tilde(i+1) y_tilde(j+1)],squared);

        for signi=-1:2:1
            for signj=-1:2:1
                kernel_tilde((N1d +signi*i)+1 , (N1d + signj*j)+1) = tmp;
            end
        end
    end
end

fft_kernel=fft2(kernel_tilde);
b=zeros(N1d^2,nsums);
for nterms=1:nsums
    fa=vec2mat(w(:,nterms),N1d);
    fa=[zeros(N1d,2*N1d);zeros(N1d,N1d) fa ];
    result=ifft2(fft_kernel.*fft2(fa));
    result= result(1:N1d,1:N1d);
    b(:,nterms)=reshape(result.',1,[]);


end


end

