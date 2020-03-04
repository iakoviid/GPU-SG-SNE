function [rep] = repulsive(Y,n,nodims)
rep=zeros(n,nodims);
if(nodims==1)
    charges=zeros(n,2);
    for i=1:n
        charges(i,1)=Y(i);
        charges(i,2)=1;
    end
    N_boxes_per_d=25;
    points_per_box=25;
    phi=compute1D(Y,charges,N_boxes_per_d,points_per_box,1,n,2);
    charges=ones(n,1);
    phi2=compute1D(Y,charges,N_boxes_per_d,points_per_box,0,n,1);
    Z=sum(phi2(:));
    Z=Z-n;
    for i=1:n
        rep(i,1)=(-phi(i,1)+Y(i)*phi(i,2))/Z;
    end
        
elseif(nodims==2)
    charges=zeros(n,3);
    for i=1:n
        charges(i,1)=Y(i,1);
        charges(i,2)=Y(i,2);
        charges(i,3)=1;
    end
    N_boxes_per_d=10;
    points_per_box=10;
    phi=compute2D(Y,charges,N_boxes_per_d,points_per_box,1,n,3);
    charges=ones(n,1);
    phi2=compute2D(Y,charges,N_boxes_per_d,points_per_box,0,n,1);
    Z=sum(phi2(:));
    Z=Z-n;
    for i=1:n
        rep(i,1)=(-phi(i,1)+Y(i,1)*phi(i,3))/Z;
        rep(i,2)=(-phi(i,2)+Y(i,2)*phi(i,3))/Z;
    end
    
    
elseif(nodims==3)
    
    charges=zeros(n,4);
    charges(:,1:3)=Y;
    charges(:,4)=ones(n,1);
    N_boxes_per_d=10;
    points_per_box=10;
    phi=compute3D(Y,charges,N_boxes_per_d,points_per_box,1,n,4);
    charges=ones(n,1);
    phi2=compute3D(Y,charges,N_boxes_per_d,points_per_box,0,n,1);
    Z=sum(phi2(:));
    Z=Z-n;
    for i=1:n
        rep(i,1)=(-phi(i,1)+Y(i,1)*phi(i,4))/Z;
        rep(i,2)=(-phi(i,2)+Y(i,2)*phi(i,4))/Z;
        rep(i,3)=(-phi(i,3)+Y(i,3)*phi(i,4))/Z;
    end    

elseif(nodims==4)
    
    charges=zeros(n,5);
    charges(:,1:4)=Y;
    charges(:,5)=ones(n,1);
    N_boxes_per_d=10;
    points_per_box=10;
    phi=compute3D(Y,charges,N_boxes_per_d,points_per_box,1,n,5);
    charges=ones(n,1);
    phi2=compute3D(Y,charges,N_boxes_per_d,points_per_box,0,n,1);
    Z=sum(phi2(:));
    Z=Z-n;
    for i=1:n
        rep(i,1)=(-phi(i,1)+Y(i,1)*phi(i,5))/Z;
        rep(i,2)=(-phi(i,2)+Y(i,2)*phi(i,5))/Z;
        rep(i,3)=(-phi(i,3)+Y(i,3)*phi(i,5))/Z;
        rep(i,4)=(-phi(i,4)+Y(i,4)*phi(i,5))/Z;
    end    
end

end

