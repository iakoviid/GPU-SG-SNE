function [b] = g2g2dnopadd2(w,N1d,x_tilde,y_tilde,squared,nsums)
kernel_tilde=zeros(N1d,N1d);
%b=zeros(N1d^2,nsums);
wc = exp( -2*pi*1i*[0:N1d-1]'/(2*N1d) );
v=zeros(N1d,N1d,nsums);
for nterms=1:nsums
    v(:,:,nterms)=vec2mat(w(:,nterms),N1d);
end
for i = 0:N1d-1
    for j =0:N1d-1
        tmp=kernel([x_tilde(1) y_tilde(1)],[x_tilde(i+1) y_tilde(j+1)],squared);
        kernel_tilde(i+1,j+1) = tmp;
        
    end
end

Kc=kernel_tilde+[zeros(N1d,1) kernel_tilde(:,N1d:-1:2)]+[zeros(1,N1d); kernel_tilde(N1d:-1:2,:)]+[zeros(1,N1d);zeros(N1d-1,1)  kernel_tilde(N1d:-1:2,N1d:-1:2)];
b=ifft2(fft2(v).*fft2(Kc));

Kc=kernel_tilde-[zeros(N1d,1) kernel_tilde(:,N1d:-1:2)]+[zeros(1,N1d); kernel_tilde(N1d:-1:2,:)]-[zeros(1,N1d);zeros(N1d-1,1)  kernel_tilde(N1d:-1:2,N1d:-1:2)];
Kc=(wc.').*Kc;
b=b+conj(wc.').*ifft2(fft2((wc.').*v).*fft2(Kc));

Kc=kernel_tilde+[zeros(N1d,1) kernel_tilde(:,N1d:-1:2)]-[zeros(1,N1d); kernel_tilde(N1d:-1:2,:)]-[zeros(1,N1d);zeros(N1d-1,1)  kernel_tilde(N1d:-1:2,N1d:-1:2)];
Kc=wc.*Kc;
b=b+conj(wc).*ifft2(fft2(wc.*v).*fft2(Kc));

Kc=kernel_tilde-[zeros(N1d,1) kernel_tilde(:,N1d:-1:2)]-[zeros(1,N1d); kernel_tilde(N1d:-1:2,:)]+[zeros(1,N1d);zeros(N1d-1,1)  kernel_tilde(N1d:-1:2,N1d:-1:2)];
Kc=wc.*(Kc.*wc.');
v=wc.*(v.*wc.');
v=conj(wc).*ifft2(fft2(v).*fft2(Kc)).*conj(wc).';
b=b+v;

b=real(b)/4;

end

