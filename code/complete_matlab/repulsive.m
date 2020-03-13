function [rep] = repulsive(Y,n,nodims)
if(nodims==1)
   width=max(Y(:))-min(Y(:));
   Nint=20;
   if width>20
    Nint=ceil(width);
   end
   rep=gradFft1D(Y,n,Nint,5);       
elseif(nodims==2)
   width=max(Y(:))-min(Y(:));
   Nint=20;
   if width>20
    Nint=ceil(width);
   end
    rep=gradFft2D(Y,n,Nint,3);
elseif(nodims==3)
   rep=gradFft3D(Y,n,12,5);
elseif(nodims==4)
    rep=gradFft4D(Y,n,6,3);
end

end

