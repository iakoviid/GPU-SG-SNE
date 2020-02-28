function [output] = checkcirc(A)
a=size(A);
n=a(1);
result1=true;
result2=true;
for(i=2:n)
    for(j=1:n)
        if(A(i,j)~=A(1,mod(j-i,n)+1))
        result1=false;
        end
    end
end
A=A';
for(i=2:n)
    for(j=1:n)
        if(A(i,j)~=A(1,mod(j-i,n)+1))
        result2=false;
        end
    end
end
output=result1 + result2;
end

