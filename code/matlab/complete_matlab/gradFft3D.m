function [rep] = gradFft3D(Y,n,N_boxes_per_d,points_per_box)
    rep=zeros(n,2);
    charges=[Y ones(n,1)];
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

    
end
