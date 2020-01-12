function [S] = kerneleval2(points)
[m,n]=size(points);
S=zeros(m,m);
for(i=1:m)
    for(j=1:m)
        S(i,j)=1/(norm(points(i,:)-points(j,:))^2+1);
    end
end


end

