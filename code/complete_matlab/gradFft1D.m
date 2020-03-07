function [rep] = gradFft1D(Y,n,N_boxes_per_d,points_per_box)
    rep=zeros(n,1);
    charges=[Y ones(n,1)];
    phi=compute1D(Y,charges,N_boxes_per_d,points_per_box,1,n,2);
    charges=ones(n,1);
    phi2=compute1D(Y,charges,N_boxes_per_d,points_per_box,0,n,1);
    Z=sum(phi2(:));
    Z=Z-n;
    for i=1:n
        rep(i,1)=(-phi(i,1)+Y(i)*phi(i,2))/Z;
    end
end

