function [b] = fftmult2(A,x)
[m , n]=size(A);
b=zeros(m,2);
b(:,1)=A*x(:,1);
b(:,2)=A*x(:,2);



end

