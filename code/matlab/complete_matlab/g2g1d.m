function [b] = g2g1d(w,N1d,x_tilde,squared,nsums)
kernel_tilde=zeros(2*N1d,1);
for i = 0:N1d-1

        tmp=kernel(x_tilde(1),x_tilde(i+1),squared);

        for signi=-1:2:1
                kernel_tilde((N1d +signi*i)+1 ) = tmp;
            
        end
    
end
fft_kernel=fft(kernel_tilde);
b=zeros(N1d,nsums);

for nterms=1:nsums
    fa=w(:,nterms);
    fa=[zeros(N1d,1); fa ];
    result=ifft(fft_kernel.*fft(fa));
    b(:,nterms)= result(1:N1d);

end

end

