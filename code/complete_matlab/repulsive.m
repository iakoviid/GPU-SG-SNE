function [rep] = repulsive(Y,n,nodims)
rep=zeros(n,nodims);
if(nodims==1)
   gradFft1D(Y,n,20,9);       
elseif(nodims==2)
    gradFft2D(Y,n,20,9);
elseif(nodims==3)
   gradFft3D(Y,n,12,5);
elseif(nodims==4)
    gradFft4D(Y,n,6,3);
end

end

