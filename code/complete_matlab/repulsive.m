function [rep] = repulsive(Y,n,nodims)
if(nodims==1)
   rep=gradFft1D(Y,n,20,9);       
elseif(nodims==2)
    rep=gradFft2D(Y,n,20,9);
elseif(nodims==3)
   rep=gradFft3D(Y,n,12,5);
elseif(nodims==4)
    rep=gradFft4D(Y,n,6,3);
end

end

