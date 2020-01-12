function [b] = mult2(A,x)
[k, n]=size(A);    
b=zeros(k,2);
    for(i=1:k)
        for(j=1:n)
            b(i,:)=b(i,:)+A(i,j)*x(j,:);
        end
    end
end
