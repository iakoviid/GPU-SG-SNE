function [rep] = avgForces(number,precent,Y,N1,scale,k)
    n=size(Y,1);
    rep=zeros(n,3);
    N2=floor(N1*precent);
    scale=(max(Y(:))-min(Y(:)))/(k*N2)*scale;
    charges=[Y ones(n,1)];
    phi=zeros(size(charges));

    for i=0:number
        phi=phi+compute3Dshift(Y,charges,N2,k,1,n,4,i*scale);
    end
    phi=phi/(number+1);
    charges=ones(n,1);
    phi2=zeros(size(charges));

    for i=0:number
        phi2=phi2+compute3Dshift(Y,charges,N2,k,0,n,1,i*scale);
    end
    phi2=phi2/(number+1);
    Z=sum(phi2(:));
    Z=Z-n;
    for i=1:n
        rep(i,1)=(-phi(i,1)+Y(i,1)*phi(i,4))/Z;
        rep(i,2)=(-phi(i,2)+Y(i,2)*phi(i,4))/Z;
        rep(i,3)=(-phi(i,3)+Y(i,3)*phi(i,4))/Z;
    end
    
end

