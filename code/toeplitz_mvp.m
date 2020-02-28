%FFT matrix-vector product of a Toeplitz matrix
p=1024;
y = randn(p,1) ;
w = randn(p,1) ;
K=toeplitz(y);

z=K*w;


a=[0;y(p:-1:2)];
B=toeplitz(a);
C2=[K B;B K];

w2=[w;zeros(p,1)];

b=C2*w2;
b_fft=ifft(fft(w2).*fft(C2(1,:))');
disp(norm(b_fft(1:p)-z));



