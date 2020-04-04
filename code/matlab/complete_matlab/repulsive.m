function [rep] = repulsive(Y,n,nodims)
if(nodims==1)
   width=max(Y(:))-min(Y(:));   
   Nint=max(20,ceil(width));

   rep=gradFft1D(Y,n,Nint,5);       
elseif(nodims==2)
   width=(max(Y(:))-min(Y(:)));
   Nint=max(50,ceil(width));
   Nint=min(240,Nint);
   %disp(Nint);
   rep=gradFft2D(Y,n,Nint,3);
elseif(nodims==3)
   width=(max(Y(:))-min(Y(:)));
   Nint=max(14,ceil(width));
   Nint=min(60,Nint);
   %disp(Nint);
   rep=gradFft3D(Y,n,Nint,3);
elseif(nodims==4)
   width=(max(Y(:))-min(Y(:)));
   Nint=max(5,ceil(width));
   Nint=min(9,Nint);
   disp(Nint);
   rep=gradFft4D(Y,n,Nint,3);
end

end

