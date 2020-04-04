function [b] = g2g1dnopadd(w,N1d,h,squared,nsums)

kernel_tilde=zeros(N1d,1);
for j = 1:N1d
        %tmp=kernel(x_tilde(1),x_tilde(j),squared);
        tmp=1/(1+((j-1)*h)^2);
        if (squared)
            tmp=tmp^2;
        end
        kernel_tilde(j)=tmp;
end


fft_kernel=fft(kernel_tilde+[0;kernel_tilde(N1d:-1:2)]);
b=ifft(fft(w,[],1).*fft_kernel,[],1);

wc = exp( -2*pi*1i*[0:N1d-1]'/(2*N1d) );
kernel_tilde=kernel_tilde-[0;kernel_tilde(N1d:-1:2)];
fft_kernel=fft(kernel_tilde.*wc);
w=w.*wc;
w=ifft(fft(w,[],1).*fft_kernel,[],1);
w=w.*conj(wc);
b=(b+real(w))/2;
end

