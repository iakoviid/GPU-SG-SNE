n=9;
m=3;
a=[];
for(i=1:m)
    a=[a S((n-(i-1)*m):-1:(n-(i)*m+1),1)'];
    a=[a S((n-(i)*m+1),2:m)];
end

for(i=1:m-1)
    a=[a S((m:-1:1),i*m+1)'];
    a=[a S(1,i*m+2:(i+1)*m)];
    
end



